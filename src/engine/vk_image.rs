use ash::{Device, vk};
use ash;

pub unsafe fn transition_image(device: &Device,
cmd: vk::CommandBuffer,
image: vk::Image,
current_layout: vk::ImageLayout,
target_layout: vk::ImageLayout){
    let image_barrier = vk::ImageMemoryBarrier2::builder()
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_access_mask(vk::AccessFlags2::MEMORY_WRITE | vk::AccessFlags2::MEMORY_READ)

        .old_layout(current_layout)
        .new_layout(target_layout)
        .subresource_range(super::vk_initializers::image_subresource_range(
            match target_layout == vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL{
                true => vk::ImageAspectFlags::DEPTH,
                false => vk::ImageAspectFlags::COLOR
            }
        ).build())
        .image(image)
        .build();

    let image_barriers = [image_barrier];
    let dependency_info = vk::DependencyInfo::builder()
        .image_memory_barriers(&image_barriers);

    device.cmd_pipeline_barrier2(cmd, &dependency_info);
}