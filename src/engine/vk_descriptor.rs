#![allow(dead_code)]

use std::collections::VecDeque;
use ash;
use ash::vk;
use log::info;

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
#[derive(Clone)]
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

    pub unsafe fn clear_descriptors(&self, device: &ash::Device){
        device.reset_descriptor_pool(self.pool, vk::DescriptorPoolResetFlags::empty())
            .expect("Unable to reset descriptor pool!");
    }

    pub unsafe fn destroy_pool(&self, device: &ash::Device){
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

pub struct DescriptorAllocatorGrowable{
    ratios: Vec<PoolSizeRatio>,
    full_pools: Vec<vk::DescriptorPool>,
    ready_pools: Vec<vk::DescriptorPool>,
    sets_per_pool: u32,
}

impl DescriptorAllocatorGrowable{

    pub fn new() -> Self{
        Self{
            ratios: vec![],
            full_pools: vec![],
            ready_pools: vec![],
            sets_per_pool: 0,
        }
    }
    pub unsafe fn init(&mut self, device: &ash::Device, initial_sets: u32, pool_ratios: Vec<PoolSizeRatio>){
        self.ratios.clear();

        for r in pool_ratios{
            self.ratios.push(r);
        }

        let new_pool = self.create_pool(device, initial_sets, &pool_ratios);

        self.sets_per_pool = initial_sets * 1.5;

        self.ready_pools.push(new_pool);
    }

    pub unsafe fn clear_pools(&mut self, device: &ash::Device){
        for p in self.ready_pools{
            device.reset_descriptor_pool(p, vk::DescriptorPoolResetFlags::empty())
                .expect("Unable to reset descriptor pools!");
        }

        for p in self.full_pools{
            device.reset_descriptor_pool(p, vk::DescriptorPoolResetFlags::empty())
                .expect("Unable to reset descriptor pools!");
            self.ready_pools.push(p);
        }

        self.full_pools.clear();
    }

    pub unsafe fn destroy_pools(&mut self, device: &ash::Device){
        for p in self.ready_pools{
            device.destroy_descriptor_pool(
                p, None
            );
        }
        self.ready_pools.clear();

        for p in self.full_pools{
            device.destroy_descriptor_pool(p, None);
        }

        self.full_pools.clear();
    }

    pub unsafe fn allocate(&mut self, device: &ash::Device, layout: vk::DescriptorSetLayout)
    -> vk::DescriptorSet{
        let mut pool_to_use = self.get_pool(device);

        let layouts = [layout];

        let alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(pool_to_use)
            .set_layouts(&layouts);

        let mut ds = vk::DescriptorSet::default();

        let result = device.allocate_descriptor_sets(
            &alloc_info
        );

        if result.is_err(){
            self.full_pools.push(pool_to_use);

            pool_to_use = self.get_pool(device);

            alloc_info.descriptor_pool = pool_to_use;

            ds = *device.allocate_descriptor_sets(
                &alloc_info
            ).expect("Unable to get descriptor set!")
                .get(0).expect("Got no descriptor sets!");
        } else{
            ds = *result.expect("Unable to get descriptor set!").get(0)
                .expect("Got no descriptor sets!");
        }

        self.ready_pools.push(pool_to_use);

        ds
    }

    fn get_pool(&mut self, device: &ash::Device) -> vk::DescriptorPool{
        let mut new_pool: vk::DescriptorPool;

        if self.ready_pools.len() != 0{
            new_pool = *self.ready_pools.get(self.ready_pools.len() - 1)
                .expect("Unable to get last descriptor pool!");
            self.ready_pools.pop();
        } else{
            new_pool = self.create_pool(device, self.sets_per_pool, &self.ratios);

            self.sets_per_pool = self.sets_per_pool * 1.5;

            if self.sets_per_pool > 4092{
                self.sets_per_pool = 4092;          }
        }

        new_pool
    }

    fn create_pool(&mut self, device: &ash::Device, set_count: u32, pool_ratios: &Vec<PoolSizeRatio>)
    -> vk::DescriptorPool{
        let mut pool_sizes: Vec<vk::DescriptorPoolSize> = vec![];
        for ratio in &pool_ratios{
            pool_sizes.push(vk::DescriptorPoolSize::builder()
                .ty(ratio.desc_type)
                .descriptor_count((ratio.ratio * (set_count as f32)) as u32)
                .build());
        }

        let pool_info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(set_count)
            .pool_sizes(pool_sizes.as_slice());

        unsafe{
        device.create_descriptor_pool(&pool_info, None)
            .expect("Unable to create descriptor pool")
        }
    }
}

pub struct DescriptorWriter{
    image_infos: VecDeque<vk::DescriptorImageInfo>,
    buffer_infos: VecDeque<vk::DescriptorBufferInfo>,
    writes: Vec<vk::WriteDescriptorSet>
}

impl DescriptorWriter{
    pub fn new() -> Self{
        Self{
            image_infos: Default::default(),
            buffer_infos: Default::default(),
            writes: vec![],
        }
    }
    pub fn write_image(&mut self, binding: i32, image: vk::ImageView,
                        sampler: vk::Sampler, layout: vk::ImageLayout,
                        desc_type: vk::DescriptorType){
        let info = vk::DescriptorImageInfo::builder()
            .sampler(sampler)
            .image_view(image)
            .image_layout(layout)
            .build();

        self.image_infos.push_back(info);

        let infos = [info];

        let write = vk::WriteDescriptorSet::builder()
            .dst_binding(binding as u32)
            .dst_set(vk::DescriptorSet::null())
            .descriptor_type(desc_type)
            .image_info(&infos)
            .build();

        self.writes.push(write);
    }

    pub fn write_buffer(&mut self, binding: i32, buffer: vk::Buffer,
                        size: u32, offset: u32, desc_type: vk::DescriptorType){
        let info = vk::DescriptorBufferInfo::builder()
            .buffer(buffer)
            .offset(vk::DeviceSize::from(offset))
            .range(vk::DeviceSize::from(size))
            .build();
        self.buffer_infos.push_back(info);

        let infos = [info];

        let write = vk::WriteDescriptorSet::builder()
            .dst_binding(binding as u32)
            .dst_set(vk::DescriptorSet::null())
            .descriptor_type(desc_type)
            .buffer_info(&infos)
            .build();

        self.writes.push(write);
    }

    pub fn clear(&mut self){
        self.image_infos.clear();
        self.writes.clear();
        self.buffer_infos.clear();
    }

    pub fn update_set(&mut self, device: &ash::Device, set: vk::DescriptorSet){

    }
}