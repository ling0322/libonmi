#include "libllm/llm.h"

#include <string.h>

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>

#include "../../third_party/nlohmann/json.hpp"
#include "libllm/context.h"
#include "libllm/dtype.h"
#include "libllm/functional.h"
#include "libllm/generator.h"
#include "libllm/model_for_generation.h"
#include "libllm/operators.h"
#include "libllm/prompt.h"
#include "libllm/tokenizer.h"
#include "libllm/whisper.h"
#include "libllm/whisper_decoder.h"
#include "lutil/error.h"
#include "lutil/ini_config.h"
#include "lutil/log.h"
#include "lutil/strings.h"
#include "lutil/zip_file.h"

using libllm::Context;
using libllm::GenerationConfig;
using libllm::Generator;
using libllm::LongType;
using libllm::ModelForGeneration;
using libllm::Prompt;
using libllm::Tokenizer;
using libllm::whisper::RecognitionResult;
using libllm::whisper::WhisperDecoder;
using libllm::whisper::WhisperModel;
using lut::IniConfig;
using json = nlohmann::json;

thread_local std::string gJsonString;
thread_local std::string gJsonErrorMessage;

constexpr char LlmConfigKey_GeneratorType[] = "generator.type";
constexpr char LlmConfigKey_WhisperLang[] = "whisper.language";
constexpr char LlmConfigValue_Sampler[] = "sampler";
constexpr char LlmConfigValue_Whisper[] = "whisper";

struct llm_model_impl_t {
  std::shared_ptr<ModelForGeneration> model;
  std::shared_ptr<Tokenizer> tokenizer;
};

struct llm_completion_impl_t {
  std::weak_ptr<ModelForGeneration> model_for_generation;
  std::shared_ptr<Generator> generator;
};

struct llm_json_impl_t {
  json jsonObject;
};

struct llm_asr_model_impl_t {
  std::shared_ptr<WhisperModel> model;
};

struct llm_asr_recognition_impl_t {
  std::shared_ptr<WhisperDecoder> decoder;
};

namespace libllm {
namespace api {

thread_local int gErrorCode = static_cast<int>(lut::ErrorCode::OK);
thread_local char gErrorMessage[512] = "";
static std::atomic<bool> gInitialized{false};

void llmSetErrorMessage(const std::string &message) {
  std::string what = message;
  if (what.size() >= sizeof(gErrorMessage)) {
    what.erase(what.begin() + sizeof(gErrorMessage) - 4, what.end());
    what += "...";
  }
  snprintf(gErrorMessage, sizeof(gErrorMessage), "%s", what.c_str());
}

void setErrorCodeAndMessage(const lut::Error &e) {
  gErrorCode = static_cast<int>(e.getCode());
  llmSetErrorMessage(e.what());
}

llmStatus_t runAndCatch(std::function<void()> &&f) {
  try {
    f();
    return LLM_OK;
  } catch (const lut::Error &e) {
    setErrorCodeAndMessage(e);
    return static_cast<llmStatus_t>(e.getCode());
  }
}

template<typename T>
T runAndCatch(std::function<T()> &&c, T default_value) {
  try {
    return c();
  } catch (const lut::Error &e) {
    setErrorCodeAndMessage(e);
    return default_value;
  }
}

Device getDeviceFromApi(int apiDevice) {
  switch (apiDevice) {
    case LLM_DEVICE_CPU:
      return Device::getCpu();
    case LLM_DEVICE_CUDA:
      return Device::getCuda();
    case LLM_DEVICE_AUTO:
      if (Device::isCudaAvailable()) {
        return Device::getCuda();
      } else {
        return Device::getCpu();
      }
    default:
      throw lut::InvalidArgError("invalid device type");
  }
}

void checkJsonKeys(json &json, std::initializer_list<std::pair<std::string_view, bool>> schema) {
  std::set<std::string_view> keys;
  for (auto &[key, value] : json.items()) {
    keys.emplace(key);
  }

  for (const auto &entry : schema) {
    std::string_view key = entry.first;
    bool required = entry.second;

    auto it = keys.find(key);
    if (required && it == keys.end()) {
      throw lut::AbortedError(lut::sprintf("json: required key \"%s\" not found", key));
    }

    if (it != keys.end()) keys.erase(it);
  }

  if (!keys.empty()) {
    throw lut::AbortedError(lut::sprintf("json: unexpected key \"%s\"", *keys.begin()));
  }
}

template<typename T>
T getValueFromJson(json &j, std::string_view key, T defaultVal) {
  T val = defaultVal;
  if (kwargsJson.contains(key)) {
    val = kwargsJson[key];
  }

  return val;
}

GenerationConfig parseGenerationConfig(json &kwargsJson) {
  GenerationConfig config;
  config.temperature = getValueFromJson<float>(kwargsJson, "temperature", 1.0);
  config.topK = getValueFromJson<int>(kwargsJson, "top_k", 50);
  config.topP = getValueFromJson<float>(kwargsJson, "top_p", 0.8);

  return config;
}

int parseGeneratorType(const std::string &name) {
  if (name == LlmConfigValue_Sampler) {
    return Generator::Sampling;
  } else if (name == LlmConfigValue_Whisper) {
    return Generator::Whisper;
  } else {
    throw lut::AbortedError("invalid generator type: " + name);
  }
}

int32_t llmErrorSetInvalidArg(const std::string &argName) {
  llmSetErrorMessage("invalid argument: " + argName);
  return LLM_ERROR_INVALID_ARG;
}

int32_t llmErrorSetAborted(const std::string &what) {
  llmSetErrorMessage(what);
  return LLM_ERROR_ABORTED;
}

int32_t llmErrorSetInsufficientBuffer() {
  llmSetErrorMessage("Insufficient buffer size.");
  return LLM_ERROR_INSUFFICIENT_BUFFER;
}

int32_t llmErrorSetEOF() {
  llmSetErrorMessage("End of file.");
  return LLM_ERROR_EOF;
}

libllm::Device parseDevice(const std::string &device) {
  if (device == "cpu") {
    return libllm::Device::getCpu();
  } else if (device == "cuda") {
    return libllm::Device::getCuda();
  } else if (device == "auto") {
    if (Device::isCudaAvailable()) {
      return Device::getCuda();
    } else {
      return Device::getCpu();
    }
  } else {
    throw lut::AbortedError("invalid device: " + device);
  }
}

}  // namespace api
}  // namespace libllm

// -- api implementation ----------

using namespace libllm;
using namespace libllm::api;

llmStatus_t llmInit(int32_t apiVersion) {
  if (!gInitialized.exchange(true)) {
    try {
      if (apiVersion != LLM_API_VERSION) throw lut::InvalidArgError("api version mismatch.");
      lut::setLogLevel(lut::LogSeverity::kINFO);
      libllm::initOperators();

      return LLM_OK;
    } catch (const lut::Error &e) {
      gInitialized = false;
      setErrorCodeAndMessage(e);
      return static_cast<llmStatus_t>(e.getCode());
      ;
    }
  }

  return LLM_OK;
}

llmStatus_t llmDestroy() {
  if (gInitialized.exchange(false)) {
    libllm::destroyOperators();
  }

  return LLM_OK;
}

const char *llmGetLastErrorMessage() {
  return gErrorMessage;
}

int32_t llm_model_init(llm_model_t *m) {
  *m = new llm_model_impl_t();
  return 0;
}

int32_t llm_model_destroy(llm_model_t *m) {
  if (!m) return llmErrorSetInvalidArg("m");

  delete *m;
  *m = nullptr;
  return 0;
}

int32_t llm_model_load(llm_model_t *m, llm_json_t *kwargs) {
  try {
    libllm::Device device;
    std::shared_ptr<lut::ZipFile> package;
    json object = (*kwargs)->jsonObject;
    for (auto &[key, value] : object.items()) {
      if (key == "filename") {
        package = lut::ZipFile::fromFile(value);
      } else if (key == "device") {
        device = parseDevice(value);
      } else {
        return llmErrorSetAborted("invalid key in options: " + key);
      }
    }

    if (!package) return llmErrorSetAborted("options.filename undefined");
    if (device.getType() == libllm::Device::kUnknown) {
      return llmErrorSetAborted("options.device undefined");
    }

    Context ctx;
    ctx.setDevice(device);
    ctx.setFloatDType(F::getDefaultFloatType(device));
    std::shared_ptr<ModelForGeneration> model = ModelForGeneration::fromPackage(ctx, package.get());

    (*m)->model = model;
  } catch (std::exception &e) {
    return llmErrorSetAborted(e.what());
  }

  return 0;
}

int32_t llm_model_get_info(llm_model_t *m, llm_json_t *info) {
  if (!m) return llmErrorSetInvalidArg("m");
  if (!info) return llmErrorSetInvalidArg("info");

  try {
    json infoJson;
    infoJson["name"] = (*m)->model->getName();
    (*info)->jsonObject = infoJson;
  } catch (std::exception &e) {
    return llmErrorSetAborted(e.what());
  }

  return 0;
}

int32_t llm_completion_init(llm_completion_t *c) {
  *c = new llm_completion_impl_t();
  return 0;
}

int32_t llm_completion_destroy(llm_completion_t *c) {
  if (!c) return llmErrorSetInvalidArg("c");
  delete *c;
  *c = nullptr;

  return 0;
}

int32_t llm_model_complete(llm_model_t *m, llm_json_t *kwargs, llm_completion_t *comp) {
  if (!m) return llmErrorSetInvalidArg("m");
  if (!kwargs) return llmErrorSetInvalidArg("kwargs");
  if (!comp) return llmErrorSetInvalidArg("comp");

  json kwargsCopy = (*kwargs)->jsonObject;
  GenerationConfig config = parseGenerationConfigAndDeleteKey(kwargsCopy);
  (*comp)->generator = SamplingGenerator::newGenerator(config, (*m)->model);
  (*comp)->model_for_generation = (*m)->model;
}

llmCompletion_t *llmCompletion_New(llmModel_t *model) {
  return runAndCatch<llmCompletion_t *>(
      [model]() {
        if (!model) throw lut::InvalidArgError("model");
        if (!model->model_for_generation) throw lut::InvalidArgError("model not initialized");

        std::unique_ptr<llmCompletion_t> comp = std::make_unique<llmCompletion_t>();
        comp->model_for_generation = model->model_for_generation;
        comp->temperature = 1.0f;
        comp->top_k = 50;
        comp->top_p = 0.8f;

        return comp.release();
      },
      nullptr);
}

llmStatus_t llmCompletion_Delete(llmCompletion_t *comp) {
  delete comp;
  return LLM_OK;
}

llmStatus_t llmCompletion_SetConfig(llmCompletion_t *comp, const char *key, const char *value) {
  return runAndCatch([comp, key, value]() {
    if (!comp) throw lut::InvalidArgError("comp");
    if (!key) throw lut::InvalidArgError("key");
    if (!value) throw lut::InvalidArgError("value");

    comp->kvConfig[key] = value;
    return LLM_OK;
  });
}

llmStatus_t llmCompletion_SetPrompt(llmCompletion_t *comp, llmPrompt_t *prompt) {
  return runAndCatch([comp, prompt]() {
    if (!comp) throw lut::InvalidArgError("comp");
    if (!prompt) throw lut::InvalidArgError("prompt");
    if (comp->generator) throw lut::InvalidArgError("completion already started");
    if (prompt->prompt->empty()) throw lut::InvalidArgError("prompt is empty");

    comp->prompt = prompt->prompt;
    return LLM_OK;
  });
}

llmStatus_t llmCompletion_SetTopP(llmCompletion_t *comp, float topP) {
  return runAndCatch([comp, topP]() {
    if (!comp) throw lut::InvalidArgError("comp");
    if (comp->generator) throw lut::InvalidArgError("completion already started");

    comp->top_p = topP;
    return LLM_OK;
  });
}

llmStatus_t llmCompletion_SetTopK(llmCompletion_t *comp, int32_t topK) {
  return runAndCatch([comp, topK]() {
    if (!comp) throw lut::InvalidArgError("comp");
    if (comp->generator) throw lut::InvalidArgError("completion already started");

    comp->top_k = topK;
    return LLM_OK;
  });
}

llmStatus_t llmCompletion_SetTemperature(llmCompletion_t *comp, float temperature) {
  return runAndCatch([comp, temperature]() {
    if (!comp) throw lut::InvalidArgError("comp");
    if (comp->generator) throw lut::InvalidArgError("completion already started");

    comp->temperature = temperature;
    return LLM_OK;
  });
}

llmBool_t llmCompletion_Next(llmCompletion_t *comp) {
  try {
    if (!comp) throw lut::InvalidArgError("comp");
    if (comp->prompt->empty()) throw lut::InvalidArgError("prompt is empty");

    if (comp->error.getCode() != lut::ErrorCode::OK) {
      return LLM_FALSE;
    }

    if (!comp->generator) {
      // prefill
      std::shared_ptr<ModelForGeneration> model = comp->model_for_generation.lock();
      if (!model) throw lut::InvalidArgError("model had been destroyed");

      GenerationConfig config;
      config.temperature = comp->temperature;
      config.topK = comp->top_k;
      config.topP = comp->top_p;

      int generatorType = Generator::Sampling;
      std::string whisperLang;
      for (const auto &kv : comp->kvConfig) {
        if (kv.first == LlmConfigKey_GeneratorType) {
          generatorType = parseGeneratorType(kv.second);
        } else if (kv.first == LlmConfigKey_WhisperLang) {
          whisperLang = lut::trim(kv.second);
        } else {
          throw lut::AbortedError("invalid configuration key: " + kv.first);
        }
      }

      if (generatorType == Generator::Sampling) {
        comp->generator = SamplingGenerator::newGenerator(config, model);
      } else {
        NOT_IMPL();
      }

      comp->generator->setPrompt(*comp->prompt);
    }

    bool ok = comp->generator->generate();
    if (ok) {
      return LLM_TRUE;
    } else {
      return LLM_FALSE;
    }
  } catch (const lut::Error &e) {
    if (comp) comp->error = e;
    return LLM_FALSE;
  }
}

llmStatus_t llmCompletion_GetError(llmCompletion_t *comp) {
  if (!comp) {
    lut::Error err = lut::InvalidArgError("comp");
    setErrorCodeAndMessage(err);
    return static_cast<llmStatus_t>(err.getCode());
  }

  if (comp->error.getCode() == lut::ErrorCode::OK) {
    return LLM_OK;
  } else {
    setErrorCodeAndMessage(comp->error);
    return static_cast<llmStatus_t>(comp->error.getCode());
  }
}

const char *llmCompletion_GetText(llmCompletion_t *comp) {
  return runAndCatch<const char *>(
      [comp]() {
        if (!comp) throw lut::InvalidArgError("comp");
        if (!comp->generator) throw lut::InvalidArgError("completion not started");

        comp->chunkText = comp->generator->getToken();
        return comp->chunkText.c_str();
      },
      nullptr);
}

const char *llmCompletion_GetToken(llmCompletion_t *comp) {
  return runAndCatch<const char *>(
      [comp]() {
        if (!comp) throw lut::InvalidArgError("comp");
        if (!comp->generator) throw lut::InvalidArgError("completion not started");

        comp->chunkText = comp->generator->getTokenName();
        return comp->chunkText.c_str();
      },
      nullptr);
}

int32_t llm_json_init(llm_json_t *j) {
  llm_json_impl_t *llmJson = new llm_json_impl_t();
  *j = llmJson;

  return 0;
}

int32_t llm_json_destroy(llm_json_t *j) {
  delete *j;
  *j = nullptr;

  return 0;
}

int32_t llm_json_parse(llm_json_t *j, const char *json_str) {
  if (!j) return llmErrorSetInvalidArg("j");
  if (!json_str) return llmErrorSetInvalidArg("json_str");

  try {
    (*j)->jsonObject = json::parse(json_str);
  } catch (lut::Error &e) {
    return llmErrorSetAborted(e.what());
  }

  return 0;
}

int32_t llm_json_dump(llm_json_t *j, char *buf, int64_t buf_size) {
  if (!j) return llmErrorSetInvalidArg("j");
  if (!buf) return llmErrorSetInvalidArg("buf");
  if (buf_size <= 0) return llmErrorSetInvalidArg("buf_size");

  std::string jsonStr = (*j)->jsonObject.dump();
  if (jsonStr.size() >= buf_size) {
    return llmErrorSetInsufficientBuffer();
  }

  snprintf(buf, buf_size, "%s", jsonStr.c_str());
  return 0;
}

int32_t llm_asr_model_init(llm_asr_model_t *m) {
  *m = new llm_asr_model_impl_t();
  return 0;
}

int32_t llm_asr_model_load(llm_asr_model_t *m, llm_json_t *options) {
  if (!m) return llmErrorSetInvalidArg("m");
  if (!options) return llmErrorSetInvalidArg("options");

  try {
    json object = (*options)->jsonObject;
    checkJsonKeys(object, {{"filename", true}, {"device", true}});

    std::shared_ptr<lut::ZipFile> package = lut::ZipFile::fromFile(object["filename"]);
    libllm::Device device = parseDevice(object["device"]);

    Context ctx = Context().withName("whisper");
    ctx.setDevice(device);
    ctx.setFloatDType(F::getDefaultFloatType(device));
    std::shared_ptr<WhisperModel> whisperModel = WhisperModel::fromPackage(ctx, package.get());

    (*m)->model = whisperModel;
  } catch (std::exception &e) {
    return llmErrorSetAborted(e.what());
  }

  return 0;
}

int32_t llm_asr_model_destroy(llm_asr_model_t *m) {
  delete *m;
  *m = nullptr;

  return 0;
}

int32_t llm_asr_recognition_init(llm_asr_recognition_t *r) {
  *r = new llm_asr_recognition_impl_t();
  return 0;
}

int32_t llm_asr_recognition_destroy(llm_asr_recognition_t *r) {
  delete *r;
  *r = nullptr;

  return 0;
}

int32_t llm_asr_recognize_media_file(
    llm_asr_model_t *model,
    llm_json_t *options,
    llm_asr_recognition_t *recognition) {
  if (!recognition) return llmErrorSetInvalidArg("r");
  if ((!model) || !(*model)->model) return llmErrorSetInvalidArg("model");
  if (!options) return llmErrorSetInvalidArg("options");

  try {
    std::string mediaFile;
    json object = (*options)->jsonObject;
    for (auto &[key, value] : object.items()) {
      if (key == "media_file") {
        mediaFile = value;
      } else {
        throw lut::AbortedError("invalid key in options: " + key);
      }
    }

    std::shared_ptr<WaveStream> stream = FFmpegWaveStream::open(mediaFile);
    std::shared_ptr<WhisperModel> whisperModel = (*model)->model;
    std::shared_ptr<Wave> wave = std::make_shared<Wave>(stream);
    std::shared_ptr<WhisperDecoder> decoder = WhisperDecoder::create(whisperModel, wave);

    (*recognition)->decoder = decoder;
  } catch (std::exception &e) {
    return llmErrorSetAborted(e.what());
  }

  return 0;
}

int32_t llm_asr_recognition_get_next_result(llm_asr_recognition_t *r, llm_json_t *result) {
  if ((!r) || !(*r)->decoder) return llmErrorSetInvalidArg("r");
  if (!result) return llmErrorSetInvalidArg("result");

  try {
    std::optional<RecognitionResult> recoResult = (*r)->decoder->nextResult();
    if (recoResult) {
      json resultJson;
      resultJson["text"] = recoResult->text;
      resultJson["language"] = recoResult->language;
      resultJson["begin"] = recoResult->begin.totalNanoseconds() / 1000000;
      resultJson["end"] = recoResult->end.totalNanoseconds() / 1000000;

      (*result)->jsonObject = resultJson;
    } else {
      return llmErrorSetEOF();
    }
  } catch (std::exception &e) {
    return llmErrorSetAborted(e.what());
  }

  return 0;
}
