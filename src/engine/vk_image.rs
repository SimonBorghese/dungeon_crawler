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

pub unsafe fn copy_image_to_image(
    device: &Device,
    cmd: vk::CommandBuffer,
    source: vk::Image,
    destination: vk::Image,
    src_size: vk::Extent2D,
    dst_size: vk::Extent2D,
){
    let subresource = vk::ImageSubresourceLayers::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_array_layer(0)
        .layer_count(1)
        .mip_level(0)
        .build();

    let blit_region = vk::ImageBlit2::builder()
        .src_offsets([
            vk::Offset3D::default(),
            vk::Offset3D::builder()
                .x(src_size.width as i32)
                .y(src_size.height as i32)
                .z(1)
                .build()])
        .dst_offsets([
            vk::Offset3D::default(),
            vk::Offset3D::builder()
                .x(dst_size.width as i32)
                .y(dst_size.height as i32)
                .z(1)
                .build()
        ])
        .src_subresource(subresource)
        .dst_subresource(subresource)
        .build();

    let regions = [blit_region];
    let blit_info = vk::BlitImageInfo2::builder()
        .dst_image(destination)
        .dst_image_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .src_image(source)
        .src_image_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .filter(vk::Filter::LINEAR)
        .regions(&regions);

    device.cmd_blit_image2(cmd, &blit_info);
}