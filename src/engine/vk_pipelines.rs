use std::io::Read;
use super::vk_initializers::*;
use ash;
use ash::vk;

pub unsafe fn load_shader_module(
    file_path: String,
    device: &ash::Device,
) -> ash::prelude::VkResult<vk::ShaderModule>{
    let mut file = std::fs::File::open(file_path)
        .expect("Unable to open file!");

    let mut file_data: Vec<u8> = vec![];
    file.read_to_end(&mut file_data)
        .expect("Unable to read file!");


    let mut create_info = vk::ShaderModuleCreateInfo::builder();
    create_info.p_code = file_data.as_ptr().cast();
    create_info.code_size = (file_data.len());

    device.create_shader_module(&create_info, None)
}