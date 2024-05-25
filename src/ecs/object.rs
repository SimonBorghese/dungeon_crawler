use crate::engine::vk_engine;
trait Object{
    fn init(engine: Option<vk_engine::VulkanEngine>);

    fn game_loop(delta_time: f32);

    fn quit();
}