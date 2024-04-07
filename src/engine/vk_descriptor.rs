use ash;
use std;
use ash::vk;

pub struct DescriptorLayoutBuilder{
    bindings: Vec<vk::DescriptorSetLayoutBinding>
}

impl DescriptorLayoutBuilder{
    pub fn new() -> Self{
        Self{
            bindings: vec![]
        }
    }
    pub fn add_binding(&mut self, binding: u32, desc_type: vk::DescriptorType){
        let new_bind =
            vk::DescriptorSetLayoutBinding::builder()
                .binding(binding)
                .descriptor_count(1)
                .descriptor_type(desc_type)
                .build();

        self.bindings.push(new_bind);
    }

    pub fn clear(&mut self){
        self.bindings.clear();
    }

    pub unsafe fn build(&mut self, device: &ash::Device, shader_stages: vk::ShaderStageFlags)
    -> vk::DescriptorSetLayout{
        for bind in &mut self.bindings{
            bind.stage_flags |= shader_stages;
        }

        let info = vk::DescriptorSetLayoutCreateInfo::builder();
        info.p_bindings = self.bindings.as_ptr();
        info.binding_count = self.bindings.len() as u32;


        device.create_descriptor_set_layout(&info, None)
            .expect("Unable to create descriptor set layout!")
    }
}

pub struct PoolSizeRatio{
    pub desc_type: vk::DescriptorType,
    pub ratio: f32
}
pub struct DescriptorAllocator{
    pub pool: vk::DescriptorPool,
}

impl DescriptorAllocator{
    pub fn new() -> Self{
        Self{
            pool: Default::default()
        }
    }

    pub fn init_pool(device: &vk::Device, max_sets: u32, pool_ratios: Vec<PoolSizeRatio>){
        let mut pool_sizes: Vec<vk::DescriptorPoolSize> = vec![];
    }

    pub fn clear_descriptors(device: &vk::Device){

    }

    pub fn destroy_pool(device: &vk::Device){

    }

    pub fn allocate(device: &vk::Device, layout: vk::DescriptorSetLayout){

    }
}