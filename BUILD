load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_library")
load("@rules_python//python:defs.bzl", "py_binary")
load("@bazel_gazelle//:def.bzl", "gazelle")

# gazelle:prefix github.com/grahamjenson/bazel-python-to-golang
gazelle(name = "gazelle")


cc_library(
    name = "maincu",
    srcs = [":main.cu"],
    copts=[
        "-x",
        "cuda",
        "-DGOOGLE_CUDA=1",
        "-Xcuda-fatbinary=--compress-all",
        "--no-cuda-include-ptx=all",
    ]
)

cc_binary(
    name = "hello",
    deps = [":maincu"]
)

# py_binary(
#     name = "main",
#     srcs = ["main.py"],
#     data = [":project"]
# )

# go_library(
#     name = "project_lib",
#     srcs = ["main.go"],
#     cgo = True,
#     importpath = "github.com/grahamjenson/bazel-python-to-golang",
# )

# go_binary(
#     name = "project",
#     embed = [":project_lib"],
#     linkmode="c-shared",
#     out="_golib.so"
# )