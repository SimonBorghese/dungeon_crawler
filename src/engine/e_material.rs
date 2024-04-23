use ash::vk;
use super::vk_engine;

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

#[derive(Clone, Default)]
pub struct MaterialInstance{
    pub pipeline: MaterialPipeline,
    pub material_set: vk::DescriptorSet,
    pub pass_type: MaterialPass
}