use std;
use image;
use image::io::Reader;

const IMAGE_PATH: &str = "assets/textures/";
pub fn load_image(raw_path: std::path::PathBuf) -> Option<image::DynamicImage>{
    let mut main_path = String::from(IMAGE_PATH);

    main_path.push_str(raw_path.file_name()
                                          .expect("Unable to get image path!").to_str()
                                          .expect("Unable to convert OSStr to Str"));

    main_path.push_str(".png");

    let true_path = std::path::PathBuf::from(main_path);

    let img_data = Reader::open(true_path);
    if img_data.is_ok(){
        let decoded_image = img_data.unwrap().decode();
        if decoded_image.is_ok(){
            return Some(decoded_image.unwrap())
        }
    }
    None
}