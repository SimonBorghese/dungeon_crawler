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

pub fn image_create_info(format: vk::Format, usage_flags: vk::ImageUsageFlags, extent: vk::Extent3D)
-> vk::ImageCreateInfoBuilder<'static>{
    vk::ImageCreateInfo::builder()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(extent)
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage_flags)
}

pub fn imageview_create_info(format: vk::Format, image: vk::Image, aspect_flags: vk::ImageAspectFlags)
-> vk::ImageViewCreateInfoBuilder<'static> {
    let resource_range = vk::ImageSubresourceRange::builder()
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1)
        .aspect_mask(aspect_flags)
        .build();

    vk::ImageViewCreateInfo::builder()
        .view_type(vk::ImageViewType::TYPE_2D)
        .image(image)
        .format(format)
        .subresource_range(resource_range)
}

pub fn attachment_info(view: vk::ImageView, clear: Option<vk::ClearValue>, layout: vk::ImageLayout)
-> vk::RenderingAttachmentInfoBuilder<'static>{
    vk::RenderingAttachmentInfo::builder()
        .image_view(view)
        .image_layout(layout)
        .load_op(vk::AttachmentLoadOp::LOAD)
        .store_op(vk::AttachmentStoreOp::STORE)
        .clear_value({
            if clear.is_some(){
                clear.unwrap()
            } else{
                vk::ClearValue::default()
            }
        })
}

pub fn depth_attachment_info(view: vk::ImageView, clear: Option<vk::ClearValue>, layout: vk::ImageLayout)
                       -> vk::RenderingAttachmentInfoBuilder<'static>{
    let mut depth_clear = vk::ClearValue::default();
    unsafe {
        depth_clear.depth_stencil.depth = 0.0;
    }
    vk::RenderingAttachmentInfo::builder()
        .image_view(view)
        .image_layout(layout)
        .load_op(vk::AttachmentLoadOp::LOAD)
        .store_op(vk::AttachmentStoreOp::STORE)
        .clear_value(depth_clear)
}

pub fn pipeline_shader_stage_create_info(shader: vk::ShaderStageFlags, module: vk::ShaderModule)
-> vk::PipelineShaderStageCreateInfoBuilder<'static>{
    vk::PipelineShaderStageCreateInfo::builder()
        .stage(shader)
        .module(module)
        .name(unsafe{std::ffi::CStr::from_ptr(b"main\0".as_ptr() as _)})
}

pub fn pipeline_layout_create_info() -> vk::PipelineLayoutCreateInfoBuilder<'static>{
    vk::PipelineLayoutCreateInfo::builder()
}