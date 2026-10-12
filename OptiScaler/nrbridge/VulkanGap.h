#pragma once

// The Vulkan half of the between slot, and only the hand-over: a device and queue on the
// game's own adapter, the two shared D3D12 fences seen from Vulkan as timeline semaphores, and
// one submit in the slot. It is a separate module because the reference keeps its interop in a
// separate module too (its pe bridge), and the host's frame code should read as frame code.
//
// Two facts about this device decide how it is created:
//
//   - it must be the adapter the game is rendering on. The reference matches the adapter LUID
//     and takes nothing from the enumeration order, and on this machine Vulkan lists the AMD
//     card first, so the first physical device is not the one the game is on.
//   - nothing here is sized. The resolutions the game hands over are whatever the game chose,
//     and with dynamic resolution they change from frame to frame; whatever is made for them is
//     made from the frame's own numbers, and made again when they change.
//
// The submit carries no work yet. What it carries is the handshake: it waits for the value the
// game's queue signalled, and signals the value the game's queue will later wait for. Both
// directions are the reference's own, in its own order -- producer->Signal, vkQueueSubmit,
// producer->Wait -- with the Wait deliberately left to the host and to the next measurement.

#include <d3d12.h>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_win32.h>
#include <wrl/client.h>

#include <cstring>
#include <string>
#include <vector>

namespace VulkanGap
{

// The reference waits on a slot's fence before recording into it again, and treats a wait that
// times out three times in a row as a lost GPU. Ours waits once, and on a timeout leaves the
// slot alone rather than recording over a command buffer the GPU may still hold.
constexpr uint64_t kSlotWaitNs = 250ull * 1000ull * 1000ull;

struct Bridge
{
    bool ready = false;
    std::string failure;

    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queueFamily = 0;
    VkCommandPool pool = VK_NULL_HANDLE;

    // A ring, for the same reason the reference has one: a command buffer cannot be recorded
    // again while the GPU is still running the last submission recorded into it.
    VkCommandBuffer cmd[3] {};
    VkFence done[3] {};
    uint32_t next = 0;

    // The hand-over. toVk is the fence the game's queue signals and Vulkan waits on; fromVk is
    // the fence Vulkan signals and the game's queue waits on. Both are D3D12 fences, shared by
    // handle and imported into Vulkan as timeline semaphores: the same import, used in opposite
    // directions, which is what makes one queue's work ordered against the other's.
    Microsoft::WRL::ComPtr<ID3D12Fence> toVk;
    Microsoft::WRL::ComPtr<ID3D12Fence> fromVk;
    VkSemaphore toVkSem = VK_NULL_HANDLE;
    VkSemaphore fromVkSem = VK_NULL_HANDLE;
};

inline Bridge& TheBridge()
{
    static Bridge b;
    return b;
}

// A shared D3D12 fence as a timeline semaphore. Verbatim from the probe that measured this on
// this adapter: the semaphore type and the import go in one create, and the handle is closed
// once the semaphore owns it.
inline VkSemaphore ImportFence(VkDevice device, HANDLE handle)
{
    VkSemaphoreTypeCreateInfo type { VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO };
    type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    type.initialValue = 0;

    VkImportSemaphoreWin32HandleInfoKHR import { VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR };
    import.pNext = &type;
    import.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE_BIT;
    import.handle = handle;

    VkSemaphoreCreateInfo ci { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &import };

    VkSemaphore semaphore = VK_NULL_HANDLE;
    if (vkCreateSemaphore(device, &ci, nullptr, &semaphore) != VK_SUCCESS)
        return VK_NULL_HANDLE;
    return semaphore;
}

inline VkSemaphore ShareAndImportFence(ID3D12Device* d3d, ID3D12Fence** out, std::string& why)
{
    Microsoft::WRL::ComPtr<ID3D12Fence> fence;
    if (FAILED(d3d->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&fence))) || !fence)
    {
        why = "could not create a shared D3D12 fence";
        return VK_NULL_HANDLE;
    }

    HANDLE handle = nullptr;
    if (FAILED(d3d->CreateSharedHandle(fence.Get(), nullptr, GENERIC_ALL, nullptr, &handle)) || !handle)
    {
        why = "could not share the D3D12 fence";
        return VK_NULL_HANDLE;
    }

    const VkSemaphore semaphore = ImportFence(TheBridge().device, handle);
    CloseHandle(handle);

    if (semaphore == VK_NULL_HANDLE)
    {
        why = "the shared fence could not be imported as a timeline semaphore";
        return VK_NULL_HANDLE;
    }

    *out = fence.Detach();
    return semaphore;
}

inline bool Create(ID3D12Device* d3d, std::string& why)
{
    Bridge& b = TheBridge();
    if (b.ready)
        return true;

    // The adapter the game is on, by LUID: see the note at the top of this file.
    const LUID luid = d3d->GetAdapterLuid();

    VkApplicationInfo app { VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app.pApplicationName = "OptiScaler";
    app.applicationVersion = 1;
    app.pEngineName = "NrBridge";
    app.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo ici { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ici.pApplicationInfo = &app;

    if (vkCreateInstance(&ici, nullptr, &b.instance) != VK_SUCCESS)
    {
        why = "vkCreateInstance failed";
        return false;
    }

    uint32_t count = 0;
    if (vkEnumeratePhysicalDevices(b.instance, &count, nullptr) != VK_SUCCESS || count == 0)
    {
        why = "no Vulkan physical device at all";
        return false;
    }

    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(b.instance, &count, devices.data());

    // Never the first device: the enumeration order is the driver's, not the game's.
    for (VkPhysicalDevice device : devices)
    {
        VkPhysicalDeviceIDProperties id { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES };
        VkPhysicalDeviceProperties2 props { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
        props.pNext = &id;
        vkGetPhysicalDeviceProperties2(device, &props);

        if (id.deviceLUIDValid && std::memcmp(id.deviceLUID, &luid, sizeof luid) == 0)
        {
            if (props.properties.apiVersion >= VK_API_VERSION_1_3)
                b.physical = device;
            break;
        }
    }

    if (b.physical == VK_NULL_HANDLE)
    {
        why = "no Vulkan device has the game adapter's LUID";
        return false;
    }

    uint32_t families = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(b.physical, &families, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(families);
    vkGetPhysicalDeviceQueueFamilyProperties(b.physical, &families, queueFamilies.data());

    for (uint32_t i = 0; i < families; ++i)
    {
        const auto flags = queueFamilies[i].queueFlags;
        if ((flags & VK_QUEUE_GRAPHICS_BIT) != 0 && (flags & VK_QUEUE_COMPUTE_BIT) != 0)
        {
            b.queueFamily = i;
            break;
        }
    }

    // Timeline semaphores, and the win32 handles the shared fences arrive as. Both are what the
    // reference requires of its device too.
    VkPhysicalDeviceVulkan12Features f12 { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    f12.timelineSemaphore = VK_TRUE;

    const char* extensions[] = { VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME,
                                 VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME };

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    queueInfo.queueFamilyIndex = b.queueFamily;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &priority;

    VkDeviceCreateInfo dci { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    dci.pNext = &f12;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &queueInfo;
    dci.enabledExtensionCount = 2;
    dci.ppEnabledExtensionNames = extensions;

    if (vkCreateDevice(b.physical, &dci, nullptr, &b.device) != VK_SUCCESS)
    {
        why = "vkCreateDevice failed";
        return false;
    }

    vkGetDeviceQueue(b.device, b.queueFamily, 0, &b.queue);

    VkCommandPoolCreateInfo poolInfo { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = b.queueFamily;

    if (vkCreateCommandPool(b.device, &poolInfo, nullptr, &b.pool) != VK_SUCCESS)
    {
        why = "vkCreateCommandPool failed";
        return false;
    }

    VkCommandBufferAllocateInfo alloc { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    alloc.commandPool = b.pool;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = 3;

    if (vkAllocateCommandBuffers(b.device, &alloc, b.cmd) != VK_SUCCESS)
    {
        why = "vkAllocateCommandBuffers failed";
        return false;
    }

    for (uint32_t i = 0; i < 3; ++i)
    {
        VkFenceCreateInfo fenceInfo { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT; // the first use of a slot has nothing to wait for
        if (vkCreateFence(b.device, &fenceInfo, nullptr, &b.done[i]) != VK_SUCCESS)
        {
            why = "vkCreateFence failed";
            return false;
        }
    }

    ID3D12Fence* to = nullptr;
    ID3D12Fence* from = nullptr;
    b.toVkSem = ShareAndImportFence(d3d, &to, why);
    if (b.toVkSem == VK_NULL_HANDLE)
        return false;
    b.toVk = to;

    why.clear();
    b.fromVkSem = ShareAndImportFence(d3d, &from, why);
    if (b.fromVkSem == VK_NULL_HANDLE)
        return false;
    b.fromVk = from;

    b.ready = true;
    return true;
}

// One submission in the slot: wait for the value the game's queue has signalled, run whatever
// is recorded (nothing yet), signal the value the game's queue will wait for. Returns a VkResult
// so the caller can decide whether the frame's Wait may be queued at all -- a Wait on a value
// nothing is going to signal would stop the game's queue, not slow it.
inline VkResult Submit(uint64_t value, std::string& why)
{
    Bridge& b = TheBridge();
    if (!b.ready)
    {
        why = "the bridge is not up";
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    const uint32_t slot = b.next % 3;
    b.next += 1;

    const VkResult waited = vkWaitForFences(b.device, 1, &b.done[slot], VK_TRUE, kSlotWaitNs);
    if (waited != VK_SUCCESS)
    {
        why = waited == VK_TIMEOUT ? "the slot's last submission has not finished" : "vkWaitForFences failed";
        return waited;
    }

    VkResult r = vkResetFences(b.device, 1, &b.done[slot]);
    if (r != VK_SUCCESS)
    {
        why = "vkResetFences failed";
        return r;
    }

    VkCommandBuffer c = b.cmd[slot];
    r = vkResetCommandBuffer(c, 0);
    if (r != VK_SUCCESS)
    {
        why = "vkResetCommandBuffer failed";
        return r;
    }

    VkCommandBufferBeginInfo begin { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    r = vkBeginCommandBuffer(c, &begin);
    if (r != VK_SUCCESS)
    {
        why = "vkBeginCommandBuffer failed";
        return r;
    }

    // Nothing recorded: the handshake is the only thing being measured. The frame's images go
    // here once they are imported, at the sizes the frame itself reports.
    r = vkEndCommandBuffer(c);
    if (r != VK_SUCCESS)
    {
        why = "vkEndCommandBuffer failed";
        return r;
    }

    VkTimelineSemaphoreSubmitInfo timeline { VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO };
    timeline.waitSemaphoreValueCount = 1;
    timeline.pWaitSemaphoreValues = &value;
    timeline.signalSemaphoreValueCount = 1;
    timeline.pSignalSemaphoreValues = &value;

    const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;

    VkSubmitInfo si { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.pNext = &timeline;
    si.waitSemaphoreCount = 1;
    si.pWaitSemaphores = &b.toVkSem;
    si.pWaitDstStageMask = &stage;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &c;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &b.fromVkSem;

    r = vkQueueSubmit(b.queue, 1, &si, b.done[slot]);
    if (r != VK_SUCCESS)
        why = "vkQueueSubmit failed";
    return r;
}

} // namespace VulkanGap
