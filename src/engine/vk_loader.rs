use super::vk_types;
use super::vk_engine;
use russimp;
use russimp::scene::PostProcess;
use std::default::Default;
use glm;

#[derive(Default, Copy, Clone)]
pub struct GeoSurface{
    pub start_index: u32,
    pub count: u32
}

#[derive(Default)]
pub struct MeshAsset{
    pub name: String,

    pub surfaces: Vec<GeoSurface>,
    pub mesh_buffers: vk_types::GPUMeshBuffers
}

impl MeshAsset{
    pub unsafe fn load_gltf_meshes(
        engine: &mut vk_engine::VulkanEngine,
        path: std::path::PathBuf
    ) -> Vec<MeshAsset>{
        println!("Loading GLTF: {}", path.file_name()
            .expect("Couldn't get file name!").to_str()
            .expect("Unable to convert to string!"));

        let scene = russimp::scene::Scene::from_file(
            path.as_os_str().to_str()
                .expect("Unable to convert path to string!"),
            vec![PostProcess::CalculateTangentSpace,
                    PostProcess::Triangulate,
                    PostProcess::JoinIdenticalVertices,
                    PostProcess::SortByPrimitiveType,
                    PostProcess::OptimizeMeshes])
            .expect("Unable to load Scene!");

        let mut meshes: Vec<MeshAsset> = vec![];
        let mut indices: Vec<i32> = vec![];
        let mut vertices: Vec<vk_types::Vertex> = vec![];

        for mesh in scene.meshes{
            let mut new_mesh: MeshAsset = Default::default();

            new_mesh.name = mesh.name.clone();

            indices.clear();
            vertices.clear();

            for v in 0..mesh.vertices.len(){
                let vertex = mesh.vertices[v];
                let normal = mesh.normals[v];

                let coords = mesh.texture_coords[v].as_ref().unwrap();

                vertices.push(
                    vk_types::Vertex{
                        position: glm::vec3(
                            vertex.x, vertex.y, vertex.z
                        ),
                        uv_x: 0.0,
                        normal: glm::vec3(
                            normal.x, normal.y, normal.z
                        ),
                        uv_y: 0.0,
                        color: glm::vec4(
                            vertex.x, vertex.y, vertex.z, 1.0
                        ),
                    }
                );
            }

            for p in mesh.faces{
                let mut new_surface: GeoSurface = Default::default();
                let index = p.0;

                new_surface.start_index = indices.len() as u32;
                new_surface.count = index.len() as u32;

                new_mesh.surfaces.push(new_surface);

                for i in index{
                    indices.push(i as i32);
                }
            }

            new_mesh.mesh_buffers = engine.upload_mesh(
                indices.as_slice(),
                vertices.as_slice()
            );

            meshes.push(new_mesh);
        }

        meshes
    }

    pub unsafe fn load_first_mesh(
        engine: &mut vk_engine::VulkanEngine,
        path: std::path::PathBuf
    ) -> Option<MeshAsset>{
        println!("Loading GLTF: {}", path.file_name()
            .expect("Couldn't get file name!").to_str()
            .expect("Unable to convert to string!"));

        let scene = russimp::scene::Scene::from_file(
            path.as_os_str().to_str()
                .expect("Unable to convert path to string!"),
            vec![PostProcess::CalculateTangentSpace,
                 PostProcess::Triangulate,
                 PostProcess::JoinIdenticalVertices,
                 PostProcess::SortByPrimitiveType,
                 PostProcess::OptimizeMeshes])
            .expect("Unable to load Scene!");

        let mut mesh_out: MeshAsset = Default::default();
        let mut indices: Vec<i32> = vec![];
        let mut vertices: Vec<vk_types::Vertex> = vec![];

        for mesh in scene.meshes{
            mesh_out.name = mesh.name.clone();

            indices.clear();
            vertices.clear();

            for v in 0..mesh.vertices.len(){
                let vertex = mesh.vertices[v];
                let normal = mesh.normals[v];
                /*
                let uvs = mesh.texture_coords.get(0)
                    .expect("Unable to load texture coords!")
                    .clone()
                    .expect("Unable to get texture coords!")
                    .get(v)
                    .expect("Unable to get vertex UVs!");

                 */

                vertices.push(
                    vk_types::Vertex{
                        position: glm::vec3(
                            vertex.x, vertex.y, vertex.z
                        ),
                        uv_x: 0.0,
                        normal: glm::vec3(
                            normal.x, normal.y, normal.z
                        ),
                        uv_y: 0.0,
                        color: glm::vec4(
                            vertex.x, vertex.y, vertex.z, 1.0
                        ),
                    }
                );
            }

            for p in mesh.faces{
                let mut new_surface: GeoSurface = Default::default();
                let index = p.0;

                new_surface.start_index = indices.len() as u32;
                new_surface.count = index.len() as u32;

                mesh_out.surfaces.push(new_surface);

                for i in index{
                    indices.push(i as i32);
                }
            }

            mesh_out.mesh_buffers = engine.upload_mesh(
                indices.as_slice(),
                vertices.as_slice()
            );

        }

        Some(mesh_out)
    }
}