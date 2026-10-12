#pragma once

// How a frame's pixels cross between the two APIs in the reference: not as a shared texture,
// but as rows in one linear buffer, which D3D12 fills with CopyTextureRegion and Vulkan reads
// through an imported handle. Both halves of that are the reference's own SharedBuffer, kept
// including the 16 MB floor, which is not a size anyone would choose: in one of its games the
// driver refused the handle (E_INVALIDARG) of shared buffers of 1.9 to 4.2 MB, for seconds to
// minutes at a time, and never of those of 7.4 MB and more.
//
// Nothing here is sized either. A footprint is made from the frame's own width, height and
// format, and made again when any of the three changes, because with dynamic resolution they
// do change, from frame to frame.

#include "VulkanGap.h"

#include <d3d12.h>
#include <vulkan/vulkan.h>
#include <wrl/client.h>

#include <algorithm>
#include <cstdio>

namespace NrBridge
{

// The texel size of the formats this host hands over. Zero for one the copy cannot be described
// without: a footprint built from the wrong size is a torn image, not an error, so the answer
// has to be "I do not know" and be refused, not a plausible default.
inline UINT BitsPerPixel(DXGI_FORMAT f)
{
    switch (f)
    {
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
    case DXGI_FORMAT_R8G8B8A8_TYPELESS:
    case DXGI_FORMAT_R10G10B10A2_UNORM:
    case DXGI_FORMAT_R11G11B10_FLOAT:
    case DXGI_FORMAT_R16G16_FLOAT:
    case DXGI_FORMAT_R16G16_UNORM:
    case DXGI_FORMAT_R32_FLOAT:
        return 32;
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
    case DXGI_FORMAT_R32G32_FLOAT:
        return 64;
    default:
        return 0;
    }
}

// A texture's top-left w x h rows, packed the way both APIs copy buffers. With dynamic
// resolution a frame copies only its subrect, the top-left of this, at the same pitch.
struct Footprint
{
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT d3d {};
    VkBufferImageCopy vk {};
    UINT64 bytes = 0;

    Footprint() = default;

    Footprint(DXGI_FORMAT format, UINT w, UINT h)
    {
        const UINT texelBytes = BitsPerPixel(format) / 8;
        if (texelBytes == 0)
            return;

        const UINT pitch =
            (w * texelBytes + D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1) & ~UINT(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1);

        d3d.Footprint = { format, w, h, 1, pitch };
        d3d.Offset = 0;

        vk.bufferOffset = 0;
        vk.bufferRowLength = pitch / texelBytes;
        vk.bufferImageHeight = h;
        vk.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        vk.imageExtent = { w, h, 1 };

        bytes = UINT64(pitch) * h;
    }

    bool valid() const { return bytes != 0; }
};

// One buffer, shared by handle, seen from both APIs.
struct SharedBuffer
{
    Microsoft::WRL::ComPtr<ID3D12Resource> resource;
    HANDLE handle = nullptr;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    UINT64 size = 0;

    bool Create(ID3D12Device* d3d, UINT64 bytes, std::string& why)
    {
        if (bytes == 0)
        {
            why = "a shared buffer of no bytes";
            return false;
        }

        // Never below 16 MB, for the reason at the top of this file.
        const UINT64 width = std::max((bytes + 65535) & ~UINT64(65535), UINT64(16) << 20);

        D3D12_HEAP_PROPERTIES hp {};
        hp.Type = D3D12_HEAP_TYPE_DEFAULT;

        D3D12_RESOURCE_DESC rd {};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width = width;
        rd.Height = rd.DepthOrArraySize = rd.MipLevels = 1;
        rd.SampleDesc.Count = 1;
        rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

        Microsoft::WRL::ComPtr<ID3D12Resource> made;
        if (FAILED(d3d->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_SHARED, &rd, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                IID_PPV_ARGS(&made))) ||
            !made)
        {
            why = "could not create the shared buffer";
            return false;
        }

        HANDLE shared = nullptr;
        if (FAILED(d3d->CreateSharedHandle(made.Get(), nullptr, GENERIC_ALL, nullptr, &shared)) || shared == nullptr)
        {
            // Seen in one of the reference's games (E_INVALIDARG). These say which of its causes
            // it was, because the HRESULT alone does not.
            D3D12_HEAP_PROPERTIES got {};
            D3D12_HEAP_FLAGS flags {};
            const HRESULT query = made->GetHeapProperties(&got, &flags);

            char text[256];
            std::snprintf(text, sizeof text,
                          "the shared buffer's handle was refused: %llu bytes, heap type %d flags 0x%X (query 0x%08lX)",
                          static_cast<unsigned long long>(width), static_cast<int>(got.Type),
                          static_cast<unsigned>(flags), static_cast<unsigned long>(query));
            why = text;
            return false;
        }

        auto& vk = VulkanGap::TheBridge();

        VkExternalMemoryBufferCreateInfo external { VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO };
        external.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT;

        VkBufferCreateInfo bi { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &external };
        bi.size = width;
        bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

        VkBuffer bufferMade = VK_NULL_HANDLE;
        if (vkCreateBuffer(vk.device, &bi, nullptr, &bufferMade) != VK_SUCCESS)
        {
            CloseHandle(shared);
            why = "vkCreateBuffer (shared) failed";
            return false;
        }

        VkMemoryRequirements req {};
        vkGetBufferMemoryRequirements(vk.device, bufferMade, &req);

        VkMemoryWin32HandlePropertiesKHR props { VK_STRUCTURE_TYPE_MEMORY_WIN32_HANDLE_PROPERTIES_KHR };
        if (vk.memoryHandleProperties(vk.device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT, shared, &props) !=
            VK_SUCCESS)
        {
            vkDestroyBuffer(vk.device, bufferMade, nullptr);
            CloseHandle(shared);
            why = "the driver would not say what the shared buffer's handle can be imported as";
            return false;
        }

        // Chained onto the allocation, which is where the specification puts an imported memory
        // object: unlike the fences, memory has no separate import call to make afterwards.
        VkMemoryDedicatedAllocateInfo dedicated { VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO };
        dedicated.buffer = bufferMade;

        VkImportMemoryWin32HandleInfoKHR import { VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR, &dedicated };
        import.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT;
        import.handle = shared;

        const UINT64 allocBytes =
            std::max<UINT64>(req.size, d3d->GetResourceAllocationInfo(0, 1, &rd).SizeInBytes);
        const uint32_t type =
            VulkanGap::MemoryType(req.memoryTypeBits & props.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (type == std::numeric_limits<uint32_t>::max())
        {
            vkDestroyBuffer(vk.device, bufferMade, nullptr);
            CloseHandle(shared);
            why = "no device-local memory type this buffer's handle can be imported into";
            return false;
        }

        VkMemoryAllocateInfo ai { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &import };
        ai.allocationSize = allocBytes;
        ai.memoryTypeIndex = type;

        VkDeviceMemory memoryMade = VK_NULL_HANDLE;
        if (vkAllocateMemory(vk.device, &ai, nullptr, &memoryMade) != VK_SUCCESS)
        {
            vkDestroyBuffer(vk.device, bufferMade, nullptr);
            CloseHandle(shared);
            why = "the import of the shared buffer failed";
            return false;
        }

        if (vkBindBufferMemory(vk.device, bufferMade, memoryMade, 0) != VK_SUCCESS)
        {
            vkFreeMemory(vk.device, memoryMade, nullptr);
            vkDestroyBuffer(vk.device, bufferMade, nullptr);
            CloseHandle(shared);
            why = "vkBindBufferMemory (shared) failed";
            return false;
        }

        resource = made;
        handle = shared;
        buffer = bufferMade;
        memory = memoryMade;
        size = width;
        return true;
    }

    void Release()
    {
        auto& vk = VulkanGap::TheBridge();

        if (buffer != VK_NULL_HANDLE)
            vkDestroyBuffer(vk.device, buffer, nullptr);
        if (memory != VK_NULL_HANDLE)
            vkFreeMemory(vk.device, memory, nullptr);
        if (handle != nullptr)
            CloseHandle(handle);
        resource.Reset();

        *this = {};
    }
};

} // namespace NrBridge
