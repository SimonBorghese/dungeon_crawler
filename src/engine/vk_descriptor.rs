use ash;
use std;
use std::cmp::max;
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

        let mut info = vk::DescriptorSetLayoutCreateInfo::builder();
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

    pub unsafe fn init_pool(&mut self, device: &ash::Device, max_sets: u32, pool_ratios: Vec<PoolSizeRatio>){
        let mut pool_sizes: Vec<vk::DescriptorPoolSize> = vec![];
        for ratio in &pool_ratios{
            pool_sizes.push(vk::DescriptorPoolSize::builder()
                .ty(ratio.desc_type)
                .descriptor_count((ratio.ratio * (max_sets as f32)) as u32)
                .build());
        }

        let pool_info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(max_sets)
            .pool_sizes(pool_sizes.as_slice());

        self.pool = device.create_descriptor_pool(&pool_info, None)
            .expect("Unable to create descriptor pool");
    }

    pub unsafe fn clear_descriptors(&mut self, device: &ash::Device){
        device.reset_descriptor_pool(self.pool, vk::DescriptorPoolResetFlags::empty())
            .expect("Unable to reset descriptor pool!");
    }

    pub unsafe fn destroy_pool(&mut self, device: &ash::Device){
        device.destroy_descriptor_pool(self.pool, None);
    }

    pub unsafe fn allocate(&mut self, device: &ash::Device, layout: vk::DescriptorSetLayout)
    -> Vec<vk::DescriptorSet>{
        let layouts = [layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.pool)
            .set_layouts(&layouts);

        device.allocate_descriptor_sets(&alloc_info)
            .expect("Unable to allocate descriptor set!")
    }
}