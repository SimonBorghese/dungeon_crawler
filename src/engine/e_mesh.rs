use std::alloc::alloc;
use std::ops::{BitOr, Deref};
use ash::{Device, vk};
use super::{vk_engine, vk_types, vk_loader, e_material};
use glm;
use glm::all;
use vk_mem::Allocator;
use crate::engine::e_material::{MaterialInstance, MaterialPipeline};
use crate::engine::vk_engine::VulkanEngine;
use crate::engine::vk_types::{AllocatedImage, VulkanObject};
use super::e_loader;

#[derive(Clone)]
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
        self.material.free(device, allocator);
    }
}

impl Mesh{
    pub fn load_entities_from_file(file: std::path::PathBuf, engine: &mut VulkanEngine) -> Vec<Self>{
        let (meshes, materials) =
            vk_loader::MeshAsset::load_gltf_meshes(engine, file);

        // Create the materials from the material descriptors
        // We only have a basic diffuse shader so everyone will use the same default shader.
        let diffuse_color_material = engine.basic_color_material_pipeline.clone();
        let mut material_instances: Vec<MaterialInstance> = vec![];

        // Iterate through each material and generate a material instance
        for mat_desc in &materials{
            if mat_desc.base_color.is_some(){
                let file_path = std::path::PathBuf::from(mat_desc.base_color.clone()
                    .unwrap());

                let image_data = e_loader::load_image(file_path)
                    .expect("Unable to load image data!");


                unsafe {
                    let loaded_image = engine.create_image_from_data(
                        image_data.as_rgba8()
                            .expect("Unable to convert RGB8")
                            .as_ptr() as _, vk::Extent3D::builder()
                            .width(image_data.width())
                            .height(image_data.height())
                            .depth(1).build(), vk::Format::R8G8B8A8_UNORM,
                        vk::ImageUsageFlags::SAMPLED, false
                    );

                    engine.immediate_submit(&|device: &ash::Device, cmd: vk::CommandBuffer| {
                        super::vk_image::transition_image(
                            device, cmd,
                            loaded_image.image, vk::ImageLayout::UNDEFINED,
                            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                        );
                    });


                    material_instances.push(
                        MaterialInstance {
                            pipeline: *diffuse_color_material.clone(),
                            material_set: Default::default(),
                            pass_type: Default::default(),
                            diffuse_image: Some(loaded_image),
                        }
                    );
                }
            } else{
                let magenta = 0xFFFF00FF;
                let black = 0xFF000000;
                let mut pixels: [u32; 16*16] = [0;16*16];
                for x in 0..16{
                    for y in 0..16{
                        let value = ((x % 2) ^ (y % 2));
                        if value == 0{
                            pixels[y*16 + x] = black;
                        } else{
                            pixels[y*16 + x] = magenta;
                        }
                    }
                }

                let error_image: AllocatedImage;
                unsafe {
                    error_image = engine.create_image_from_data(
                        pixels.as_ptr() as *const _, vk::Extent3D::builder()
                            .width(16)
                            .height(16)
                            .depth(1)
                            .build(), vk::Format::B8G8R8A8_UNORM,
                        vk::ImageUsageFlags::SAMPLED, false
                    );

                    engine.immediate_submit(&|device: &ash::Device, cmd: vk::CommandBuffer| {
                        super::vk_image::transition_image(
                            device, cmd,
                            error_image.image, vk::ImageLayout::UNDEFINED,
                            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                        );
                    });
                }

                material_instances.push(
                    MaterialInstance{
                        pipeline: *diffuse_color_material.clone(),
                        material_set: Default::default(),
                        pass_type: Default::default(),
                        diffuse_image: Some(error_image),
                    }
                );
            }
        }

        let mut entities: Vec<super::e_mesh::Mesh> = vec![];

        for i in 0..meshes.len(){
            entities.push(
                super::e_mesh::Mesh{
                    mesh: meshes[i].clone(),
                    transform: num::one(),
                    engine: engine.device.clone().unwrap(),
                    material: Box::new(material_instances[i].clone()),
                }
            )
        }

        if material_instances.len() > meshes.len(){
            for i in meshes.len() .. material_instances.len(){
                unsafe {
                    material_instances[i].free(engine.device.as_ref().unwrap(),
                                               engine.allocator.as_ref().unwrap());
                }
            }
        }

        entities
    }

    pub unsafe fn bind_material(&self, device: &ash::Device, cmd: vk::CommandBuffer, image_set: vk::DescriptorSet){
        self.material.bind_material(device, cmd, image_set);
    }

    pub unsafe fn update_material(&self, device: &ash::Device, cmd: vk::CommandBuffer, image_set: vk::DescriptorSet){
        self.material.update_material(device, cmd, image_set);
    }
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