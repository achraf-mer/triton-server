#include <pybind11/pybind11.h>
#include "http_server.h"
#include "server_interface.h"

namespace py = pybind11;

// Macro used by PyWrapper
#define DISALLOW_COPY(TypeName) TypeName(const TypeName&) = delete;
#define DISALLOW_ASSIGN(TypeName) void operator=(const TypeName&) = delete;
#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
  DISALLOW_COPY(TypeName)                  \
  DISALLOW_ASSIGN(TypeName)

template <typename TritonStruct>
class PyWrapper {
 public:
  explicit PyWrapper(TritonStruct* triton_object, bool owned)
      : triton_object_(triton_object), owned_(owned)
  {
  }
  PyWrapper() = default;
  TritonStruct* Ptr() { return triton_object_; } // Get ptr of underlying Triton Object
  DISALLOW_COPY_AND_ASSIGN(PyWrapper);

 protected:
  TritonStruct* triton_object_{nullptr};
  bool owned_{false};
};





// base exception for all Triton error code
struct TritonError : public std::runtime_error {
  explicit TritonError(const std::string& what) : std::runtime_error(what) {}
};

// triton::core::python exceptions map 1:1 to TRITONSERVER_Error_Code.
struct UnknownError : public TritonError {
  explicit UnknownError(const std::string& what) : TritonError(what) {}
};
struct InternalError : public TritonError {
  explicit InternalError(const std::string& what) : TritonError(what) {}
};
struct NotFoundError : public TritonError {
  explicit NotFoundError(const std::string& what) : TritonError(what) {}
};
struct InvalidArgumentError : public TritonError {
  explicit InvalidArgumentError(const std::string& what) : TritonError(what) {}
};
struct UnavailableError : public TritonError {
  explicit UnavailableError(const std::string& what) : TritonError(what) {}
};
struct UnsupportedError : public TritonError {
  explicit UnsupportedError(const std::string& what) : TritonError(what) {}
};
struct AlreadyExistsError : public TritonError {
  explicit AlreadyExistsError(const std::string& what) : TritonError(what) {}
};

TRITONSERVER_Error*
CreateTRITONSERVER_ErrorFrom(const py::error_already_set& ex)
{
  // Reserved lookup to get Python type of the exceptions,
  // 'TRITONSERVER_ERROR_UNKNOWN' is the fallback error code.
  // static auto uk =
  // py::module::import("triton_bindings").attr("UnknownError");
  static auto it = py::module::import("tritonfrontend").attr("InternalError");
  static auto nf = py::module::import("tritonfrontend").attr("NotFoundError");
  static auto ia =
      py::module::import("tritonfrontend").attr("InvalidArgumentError");
  static auto ua =
      py::module::import("tritonfrontend").attr("UnavailableError");
  static auto us =
      py::module::import("tritonfrontend").attr("UnsupportedError");
  static auto ae =
      py::module::import("tritonfrontend").attr("AlreadyExistsError");
  TRITONSERVER_Error_Code code = TRITONSERVER_ERROR_UNKNOWN;
  if (ex.matches(it.ptr())) {
    code = TRITONSERVER_ERROR_INTERNAL;
  } else if (ex.matches(nf.ptr())) {
    code = TRITONSERVER_ERROR_NOT_FOUND;
  } else if (ex.matches(ia.ptr())) {
    code = TRITONSERVER_ERROR_INVALID_ARG;
  } else if (ex.matches(ua.ptr())) {
    code = TRITONSERVER_ERROR_UNAVAILABLE;
  } else if (ex.matches(us.ptr())) {
    code = TRITONSERVER_ERROR_UNSUPPORTED;
  } else if (ex.matches(ae.ptr())) {
    code = TRITONSERVER_ERROR_ALREADY_EXISTS;
  }
  return TRITONSERVER_ErrorNew(code, ex.what()); // TRITONSERVER_ErrorNew in tritonserver.cc
}

void
ThrowIfError(TRITONSERVER_Error* err)
{
  if (err == nullptr) {
    return;
  }
  std::shared_ptr<TRITONSERVER_Error> managed_err(
      err, TRITONSERVER_ErrorDelete);
  std::string msg = TRITONSERVER_ErrorMessage(err);
  switch (TRITONSERVER_ErrorCode(err)) {
    case TRITONSERVER_ERROR_INTERNAL:
      throw InternalError(std::move(msg));
    case TRITONSERVER_ERROR_NOT_FOUND:
      throw NotFoundError(std::move(msg));
    case TRITONSERVER_ERROR_INVALID_ARG:
      throw InvalidArgumentError(std::move(msg));
    case TRITONSERVER_ERROR_UNAVAILABLE:
      throw UnavailableError(std::move(msg));
    case TRITONSERVER_ERROR_UNSUPPORTED:
      throw UnsupportedError(std::move(msg));
    case TRITONSERVER_ERROR_ALREADY_EXISTS:
      throw AlreadyExistsError(std::move(msg));
    default:
      throw UnknownError(std::move(msg));
  }
}


class HttpServer : public PyWrapper<struct TRITONSERVER_Server> {
    public:
        PyServer()
}

// NEED TO ADD FUNCTION THAT EXPOSES C_PTR: get_c_ptr() to Server object
// [fixme] module name
PYBIND11_MODULE(tritonfrontend, m) {
    m.doc() = "Python bindings for Triton Inference Server Frontend Endpoints";
    
    auto tfe = pybind11::register_exception<TritonError>(m, "TritonError");
    pybind11::register_exception<UnknownError>(m, "UnknownError", tfe.ptr());
    pybind11::register_exception<InternalError>(m, "InternalError", tfe.ptr());
    pybind11::register_exception<NotFoundError>(m, "NotFoundError", tfe.ptr());
    pybind11::register_exception<InvalidArgumentError>(m, "InvalidArgumentError", tfe.ptr());
    pybind11::register_exception<UnavailableError>(m, "UnavailableError", tfe.ptr());
    pybind11::register_exception<UnsupportedError>(m, "UnsupportedError", tfe.ptr());
    pybind11::register_exception<AlreadyExistsError>(m, "AlreadyExistsError", tfe.ptr());

    py::module_ tritonserver = py::module_::import("tritonserver");
    py::object CoreEnum = tritonserver.attr("TRITONSERVER_ServerOptions");


    

}
