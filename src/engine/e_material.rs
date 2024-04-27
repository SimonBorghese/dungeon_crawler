use ash::{Device, vk};
use vk_mem::Allocator;
use crate::engine::vk_types::VulkanObject;
use super::vk_engine;
use super::vk_types;

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
    pub pass_type: MaterialPass
}

impl MaterialInstance{
    pub unsafe fn bind_material(&self, device: &ash::Device, cmd: vk::CommandBuffer){
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS,
        self.pipeline.pipeline);
    }
}

impl VulkanObject for MaterialInstance{
    unsafe fn free(&self, device: &Device, allocator: &Allocator) {
    }
}