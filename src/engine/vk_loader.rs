use super::vk_types;
use super::vk_engine;
use gltf;

struct GeoSurface{
    start_index: u32,
    count: u32
}

#[derive(Default)]
pub struct MeshAsset{
    name: String,

    surfaces: Vec<GeoSurface>,
    mesh_buffers: vk_types::GPUMeshBuffers
}

impl MeshAsset{
    pub fn load_gltf_meshes(
        engine: &vk_engine::VulkanEngine,
        path: std::path::PathBuf
    ) -> Option<Vec<MeshAsset>>{
        println!("Loading GLTF: {}", path.file_name()
            .expect("Couldn't get file name!").to_str()
            .expect("Unable to convert to string!"));

        let gltf = gltf::import(path)
            .expect("Unable to load GLTF File!");

        let mut meshes: Vec<MeshAsset> = vec![];
        let mut indices: Vec<u32> = vec![];
        let mut vertices: Vec<vk_types::Vertex> = vec![];

        for mesh in gltf.0.meshes(){
            let mut new_mesh: MeshAsset = Default::default();

            new_mesh.name = String::from(mesh.name()
                .expect("Unable to get mesh name!"));

            for p in mesh.primitives(){
                
            }
        }

        None
    }
}