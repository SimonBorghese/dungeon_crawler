use ash::vk;


pub fn command_pool_create_info(queue_family_index: u32, flags: vk::CommandPoolCreateFlags)
-> vk::CommandPoolCreateInfoBuilder<'static>{
    vk::CommandPoolCreateInfo::builder()
        .queue_family_index(queue_family_index)
        .flags(flags)
}

pub fn command_buffer_allocate_info(pool: vk::CommandPool, count: u32)
-> vk::CommandBufferAllocateInfoBuilder<'static>{
    vk::CommandBufferAllocateInfo::builder()
        .command_pool(pool)
        .command_buffer_count(count)
        .level(vk::CommandBufferLevel::PRIMARY)
}

pub fn fence_create_info(flags: vk::FenceCreateFlags)
-> vk::FenceCreateInfoBuilder<'static>{
    vk::FenceCreateInfo::builder()
        .flags(flags)
}

pub fn semaphore_create_info()
-> vk::SemaphoreCreateInfoBuilder<'static>{
    vk::SemaphoreCreateInfo::builder()
}

pub fn command_buffer_begin_info(flags: vk::CommandBufferUsageFlags)
-> vk::CommandBufferBeginInfoBuilder<'static>{
    vk::CommandBufferBeginInfo::builder()
        .flags(flags)
}

pub fn image_subresource_range(aspect_mask: vk::ImageAspectFlags)
-> vk::ImageSubresourceRangeBuilder<'static>{
    vk::ImageSubresourceRange::builder()
        .aspect_mask(aspect_mask)
        .base_mip_level(0)
        .level_count(vk::REMAINING_MIP_LEVELS)
        .base_array_layer(0)
        .layer_count(vk::REMAINING_ARRAY_LAYERS)
}

pub fn semaphore_submit_info(stage_mask: vk::PipelineStageFlags2, semaphore: vk::Semaphore)
-> vk::SemaphoreSubmitInfoBuilder<'static>{
    vk::SemaphoreSubmitInfo::builder()
        .semaphore(semaphore)
        .stage_mask(stage_mask)
        .device_index(0)
        .value(1)
}

pub fn command_buffer_submit_info(cmd: vk::CommandBuffer)
-> vk::CommandBufferSubmitInfoBuilder<'static>{
    vk::CommandBufferSubmitInfo::builder()
        .command_buffer(cmd)
        .device_mask(0)
}

pub fn submit_info<'a>(
    cmd: &'a [vk::CommandBufferSubmitInfo], signal_semaphore: &'a [vk::SemaphoreSubmitInfo],
    wait_semaphore_info: &'a [vk::SemaphoreSubmitInfo]
) -> vk::SubmitInfo2Builder<'a>{
    vk::SubmitInfo2::builder()
        .wait_semaphore_infos(wait_semaphore_info)
        .signal_semaphore_infos(signal_semaphore)
        .command_buffer_infos(cmd)
}
