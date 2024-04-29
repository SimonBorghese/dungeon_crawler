use ash::{Device, vk};
use vk_mem::Allocator;
use crate::engine::vk_descriptor::DescriptorWriter;
use crate::engine::vk_types::VulkanObject;
use super::vk_engine;
use super::vk_types;
use super::vk_image;

#[derive(Clone)]
pub enum MaterialPass{
    MainColor,
}

impl Default for MaterialPass{
    fn default() -> Self {
        MaterialPass::MainColor
    }
}

#[derive(Clone, Default)]
pub struct MaterialPipeline{
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout
}

impl VulkanObject for MaterialPipeline{
    unsafe fn free(&self, device: &Device, allocator: &Allocator) {
        device.destroy_pipeline_layout(self.layout, None);
        device.destroy_pipeline(self.pipeline, None);
    }
}

#[derive(Clone, Default)]
pub struct MaterialInstance{
    pub pipeline: MaterialPipeline,
    pub material_set: vk::DescriptorSet,
    pub pass_type: MaterialPass,
    pub diffuse_image: Option<vk_types::AllocatedImage>,
}

impl MaterialInstance{
    pub unsafe fn update_material(&self, device: &ash::Device, cmd: vk::CommandBuffer,
    set: vk::DescriptorSet){

        let mut writer = DescriptorWriter::new();
        writer.clear();
        writer.write_image(0, self.diffuse_image.as_ref().unwrap().image_view,
                           self.diffuse_image.as_ref().unwrap().sampler,
                           vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                           vk::DescriptorType::COMBINED_IMAGE_SAMPLER);
        writer.update_set(&device, set);
    }

    pub unsafe fn bind_material(&self, device: &ash::Device, cmd: vk::CommandBuffer,
                                  set: vk::DescriptorSet){
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS,
                                 self.pipeline.pipeline);


    }
}

impl VulkanObject for MaterialInstance{
    unsafe fn free(&self, device: &Device, allocator: &Allocator) {
        if self.diffuse_image.is_some() {
            self.diffuse_image.as_ref().unwrap().free(device, allocator);
        }
    }
}