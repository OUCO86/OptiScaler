#pragma once

// The bridge between OptiScaler's upscaler call and our own neural renderer.
//
// OptiScaler is the host and we do not own it: it is GPL-3.0 and this file must never include
// anything from it beyond what is shown here. What crosses this header is a description of one
// frame -- handles and numbers, never pixels -- so the far side can be a separate process with
// its own licence.
//
// Stage one is a probe: it reads what the host offers and writes it to the log, and changes
// nothing. That is deliberate. The whole reason for this host is motion vectors, and "the log
// says LowResMV: true" is not the same claim as "a resource arrived, at this size, with this
// scale". The probe turns the second claim into something printed, before any of our own code
// is allowed to touch a pixel.

#include <d3d12.h>
#include <nvsdk_ngx.h>

namespace NrBridge
{
// One frame, as the upscaler is about to consume it.
//
// motionScaleX/Y is the number that turns a motion vector into pixels -- it is not generally
// the frame width, and its second component is not generally positive. Everything downstream
// that reads a vector has to go through it.
struct Frame
{
    ID3D12Resource* colour = nullptr;
    ID3D12Resource* motion = nullptr;
    ID3D12Resource* depth = nullptr;
    ID3D12Resource* exposure = nullptr;

    UINT width = 0, height = 0;

    float motionScaleX = 1.0f, motionScaleY = 1.0f;
    float jitterX = 0.0f, jitterY = 0.0f;
    float preExposure = 1.0f;

    // Absent is not the same as default. A field we could not read stays false and the far side
    // must fall back rather than assume; a wrong guess here is a temporal artefact, which is
    // exactly what this project exists to remove.
    bool hasMotion = false, hasDepth = false, hasColour = false, hasExposure = false;
    bool motionJittered = false;
    bool depthInverted = false;
    bool reset = false;
};

// Read what this evaluate offers and log it. Touches no resource and records no command.
// Safe to call on every evaluate; it prints the first few frames and then only occasionally.
void ProbeBeforeUpscale(ID3D12GraphicsCommandList* cmdList, NVSDK_NGX_Parameter* params);
} // namespace NrBridge
