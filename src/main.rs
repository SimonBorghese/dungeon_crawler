use std::ops::{Add, Sub};
use sdl2::event::Event;

mod engine;

fn main(){
    let mut engine = engine::vk_engine::VulkanEngine::new(800,600);
    engine.init();

    let mut main_mesh: Vec<engine::e_mesh::Mesh>;
    let mesh: u32;
    let cube: u32;
    main_mesh = engine::e_mesh::Mesh::load_entities_from_file(
        std::path::PathBuf::from("assets/test.glb"), &mut engine
    );

    mesh = engine.add_entity(main_mesh.first()
        .expect("Unable to get first mesh!").clone());

    main_mesh = engine::e_mesh::Mesh::load_entities_from_file(
        std::path::PathBuf::from("assets/cube.glb"), &mut engine
    );

    cube = engine.add_entity(main_mesh.first()
        .expect("Unable to get first mesh!").clone());

    while !engine.run(){
        let mut transform: glm::Mat4 = num::one();
        transform = glm::ext::translate(
            &transform, glm::vec3(0.0, 0.0, -5.0 + -10.0 * glm::abs(glm::sin(engine.frame_number as f32 / 120f32)))
        );
        engine.set_entity_transform(mesh, transform);
        transform = num::one();
        transform = glm::ext::translate(
            &transform, glm::vec3(-10.0 * glm::abs(glm::sin(engine.frame_number as f32 / 120f32)), 0.0, -10.0)
            );
        engine.set_entity_transform(cube, transform);

        let keyboard_state = engine.event.keyboard_state();
        if keyboard_state.is_scancode_pressed(sdl2::keyboard::Scancode::W){
            engine.camera_position = engine.camera_position.add(
                engine.camera_forward
            );
        }
        if keyboard_state.is_scancode_pressed(sdl2::keyboard::Scancode::S){
            engine.camera_position = engine.camera_position.sub(
                engine.camera_forward
            );
        }
    }
    unsafe {
        engine.prepare_cleanup();
        engine.free_entity(cube);
        engine.free_entity(mesh);
        engine.cleanup();
    }
}
