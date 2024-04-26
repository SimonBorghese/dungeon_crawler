use std::ops::Deref;
use ash::{Device, vk};
use super::{vk_engine, vk_types, vk_loader, e_material};
use glm;
use glm::all;
use vk_mem::Allocator;
use crate::engine::vk_types::VulkanObject;

pub struct Mesh{
    pub mesh: vk_loader::MeshAsset,
    pub transform: glm::Mat4,
    pub engine: ash::Device,
    pub material: Box<e_material::MaterialInstance>
}

impl VulkanObject for Mesh{

    unsafe fn free(&self, device: &Device, allocator: &Allocator) {
        println!("Freeing Mesh!");
        self.mesh.mesh_buffers.free(device, allocator);
    }
}

impl Mesh{
    pub unsafe fn draw(&self, cmd: vk::CommandBuffer){
        let device = self.engine.clone();
        let push_constants= vk_types::GPUDrawPushConstants{
            world_matrix: self.transform,
            vertex_buffer: self.mesh.mesh_buffers.vertex_buffer_address,
        };

        device.cmd_push_constants(cmd, self.material.pipeline.layout,
                                             vk::ShaderStageFlags::VERTEX, 0, std::slice::from_raw_parts(
                &push_constants as *const _ as *const u8,
                std::mem::size_of::<vk_types::GPUDrawPushConstants>()
            ));

        device.cmd_bind_index_buffer(
            cmd, self.mesh.mesh_buffers.index_buffer.buffer,
            vk::DeviceSize::from(0u32), vk::IndexType::UINT32
        );


        for surface in &self.mesh.surfaces {
            device.cmd_draw_indexed(cmd, surface.count, 1,
                                               surface.start_index, 0, 0);
        }
    }
}