mod engine;

fn main(){
    let mut engine = engine::vk_engine::VulkanEngine::new(800,600);
    engine.init();
    while !engine.run(){

    }
    unsafe {
        engine.cleanup();
    }
}
