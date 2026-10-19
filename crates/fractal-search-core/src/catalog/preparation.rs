use super::*;
use crate::{
    preparation_pool::{
        PreparationPool, Ticket, IMAGE_RESERVATION, MEDIA_RESERVATION, TEXT_RESERVATION,
    },
    preprocess,
};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

#[derive(Debug)]
pub(super) struct FatalPreparation;
impl std::fmt::Display for FatalPreparation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Prepare indexing inputs and thumbnails")
    }
}

pub(super) enum FileData {
    Image {
        image: RgbImage,
        inputs: Option<preprocess::ImageInputs>,
        thumbnail: Vec<u8>,
    },
    Media {
        info: media::MediaInfo,
        specs: Option<Vec<media::SegmentSpec>>,
        first: Option<PreparedSegment>,
    },
}
pub(super) struct PreparedFile {
    pub stamp: Stamp,
    pub data: FileData,
}
pub(super) type FileTicket = Ticket<Result<PreparedFile>>;

pub(super) fn submit_file(
    pool: &PreparationPool,
    path: &Path,
    native: bool,
    tokenizer: Option<Arc<preprocess::QueryTokenizer>>,
) -> Result<FileTicket> {
    let kind = media::kind_for_path(path).context("Unsupported preparation file")?;
    let path = path.to_owned();
    let budget = match kind {
        MediaKind::Image => IMAGE_RESERVATION,
        MediaKind::Text => TEXT_RESERVATION,
        MediaKind::Audio | MediaKind::Video => MEDIA_RESERVATION,
    };
    pool.submit(budget, move |stop| {
        ensure!(!stop.load(Ordering::Acquire), "Index preparation cancelled");
        let before = stamp(&path)?;
        let data = if kind == MediaKind::Image {
            crate::initialize_image_decoders();
            let image = decode_image(&path)?;
            let inputs = if native {
                Some(
                    preprocess::prepare_images(std::slice::from_ref(&image))
                        .context(FatalPreparation)?,
                )
            } else {
                None
            };
            let thumbnail = thumbnail(&image).context(FatalPreparation)?;
            FileData::Image {
                image,
                inputs,
                thumbnail,
            }
        } else {
            let info = media::inspect_with_cancel(&path, &|| stop.load(Ordering::Acquire))?;
            let specs = if info.kind == MediaKind::Text && native {
                let tokenizer = tokenizer
                    .as_ref()
                    .context("Native preparation tokenizer missing")
                    .context(FatalPreparation)?;
                let mut failed = false;
                let result = media::segments_with_token_counter(&path, &info, |title, text| {
                    ensure!(!stop.load(Ordering::Acquire), "Text preparation cancelled");
                    let result = tokenizer.document_token_count(title, text);
                    failed |= result.is_err();
                    result
                });
                Some(if failed {
                    result.context(FatalPreparation)?
                } else {
                    result?
                })
            } else if info.kind != MediaKind::Text {
                Some(media::segments(&path, &info)?)
            } else {
                None
            };
            let first = specs
                .as_ref()
                .and_then(|specs| specs.first())
                .map(|spec| prepare_segment(&path, &info, spec, native, tokenizer.clone(), stop))
                .transpose()?;
            FileData::Media { info, specs, first }
        };
        ensure!(
            stamp(&path)? == before,
            "File changed during preparation: {path:?}"
        );
        Ok(PreparedFile {
            stamp: before,
            data,
        })
    })
}

pub(super) enum NativeInput {
    Raw,
    Text(preprocess::TextInputs),
    Audio(preprocess::AudioInputs),
    Video(Vec<preprocess::ImageInputs>),
}
pub(super) struct PreparedSegment {
    pub raw: PreparedMedia,
    pub input: NativeInput,
    pub thumbnail: Option<Vec<u8>>,
}

pub(super) fn submit_segment(
    pool: &PreparationPool,
    path: &Path,
    info: &media::MediaInfo,
    spec: &media::SegmentSpec,
    native: bool,
    tokenizer: Option<Arc<preprocess::QueryTokenizer>>,
) -> Result<Ticket<Result<PreparedSegment>>> {
    let path = path.to_owned();
    let info = info.clone();
    let spec = spec.clone();
    pool.submit(
        if spec.kind == MediaKind::Text {
            TEXT_RESERVATION
        } else {
            MEDIA_RESERVATION
        },
        move |stop| prepare_segment(&path, &info, &spec, native, tokenizer, stop),
    )
}

fn prepare_segment(
    path: &Path,
    info: &media::MediaInfo,
    spec: &media::SegmentSpec,
    native: bool,
    tokenizer: Option<Arc<preprocess::QueryTokenizer>>,
    stop: &AtomicBool,
) -> Result<PreparedSegment> {
    let raw =
        media::decode_segment_with_cancel(&path, &info, &spec, &|| stop.load(Ordering::Acquire))?;
    ensure!(!stop.load(Ordering::Acquire), "Index preparation cancelled");
    let input = if !native {
        NativeInput::Raw
    } else {
        match &raw {
            PreparedMedia::Text { title, text } => NativeInput::Text(
                tokenizer
                    .context("Preparation tokenizer missing")
                    .context(FatalPreparation)?
                    .encode_document(title, text)
                    .context(FatalPreparation)?,
            ),
            PreparedMedia::Audio { samples } => {
                NativeInput::Audio(preprocess::prepare_audio(samples).context(FatalPreparation)?)
            }
            PreparedMedia::Video { frames } => {
                let mut inputs = Vec::with_capacity(frames.len());
                for frame in frames {
                    ensure!(!stop.load(Ordering::Acquire), "Index preparation cancelled");
                    inputs.push(
                        preprocess::prepare_video_frames(std::slice::from_ref(frame))
                            .context(FatalPreparation)?,
                    );
                }
                NativeInput::Video(inputs)
            }
        }
    };
    let thumbnail = match &raw {
        PreparedMedia::Video { frames } => Some(
            thumbnail(
                frames
                    .first()
                    .context("Decoded video segment has no frames")?,
            )
            .context(FatalPreparation)?,
        ),
        PreparedMedia::Text { .. } | PreparedMedia::Audio { .. } => None,
    };
    Ok(PreparedSegment {
        raw,
        input,
        thumbnail,
    })
}

pub(super) fn combine_images(
    inputs: Vec<preprocess::ImageInputs>,
) -> Result<preprocess::ImageInputs> {
    ensure!(
        !inputs.is_empty() && inputs.len() <= MAX_IMAGE_BATCH_SIZE,
        "Invalid prepared image batch"
    );
    for input in &inputs {
        ensure!(
            input.max_patches == preprocess::MAX_PATCHES
                && input.num_soft_tokens.len() == 1
                && input.spatial_shapes.len() == 1
                && input.pixel_values.len() == input.max_patches * preprocess::PATCH_DIM
                && input.pixel_position_ids.len() == input.max_patches * 2,
            "Prepared image violates the pinned patch contract"
        );
    }
    // The default batch of one transfers ownership without copying ~7.4 MiB.
    // Larger batches grow once, then move the remaining prepared inputs in order.
    let additional = inputs.len() - 1;
    let mut inputs = inputs.into_iter();
    let mut combined = inputs.next().context("Missing prepared image")?;
    combined
        .pixel_values
        .reserve_exact(additional * combined.max_patches * preprocess::PATCH_DIM);
    combined
        .pixel_position_ids
        .reserve_exact(additional * combined.max_patches * 2);
    combined.num_soft_tokens.reserve_exact(additional);
    combined.spatial_shapes.reserve_exact(additional);
    for mut input in inputs {
        combined.pixel_values.append(&mut input.pixel_values);
        combined
            .pixel_position_ids
            .append(&mut input.pixel_position_ids);
        combined.num_soft_tokens.append(&mut input.num_soft_tokens);
        combined.spatial_shapes.append(&mut input.spatial_shapes);
    }
    Ok(combined)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_image_batch_reuses_buffers_and_rejects_malformed_inputs() {
        let image = RgbImage::from_pixel(53, 97, image::Rgb([12, 78, 200]));
        let input = preprocess::prepare_images(&[image]).unwrap();
        let pixels = input.pixel_values.as_ptr();
        let positions = input.pixel_position_ids.as_ptr();
        let mut combined = combine_images(vec![input]).unwrap();
        assert_eq!(combined.pixel_values.as_ptr(), pixels);
        assert_eq!(combined.pixel_position_ids.as_ptr(), positions);
        combined.pixel_values.pop();
        assert!(combine_images(vec![combined]).is_err());
        assert!(combine_images(Vec::new()).is_err());
    }

    #[test]
    fn worker_thumbnail_matches_original_pixels_exactly() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("pattern.png");
        let image = RgbImage::from_fn(613, 417, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8])
        });
        image.save(&path).unwrap();
        let mut pool = PreparationPool::new().unwrap();
        let prepared = submit_file(&pool, &path, false, None)
            .unwrap()
            .wait(&|| false)
            .unwrap()
            .unwrap()
            .unwrap();
        let FileData::Image {
            thumbnail: bytes, ..
        } = prepared.data
        else {
            panic!("Expected prepared image");
        };
        assert_eq!(bytes, thumbnail(&image).unwrap());
        pool.finish().unwrap();
    }

    #[test]
    fn merging_worker_images_matches_native_batch_preparation_exactly() {
        let images = vec![
            RgbImage::from_pixel(53, 97, image::Rgb([12, 78, 200])),
            RgbImage::from_pixel(137, 71, image::Rgb([211, 27, 41])),
        ];
        let expected = preprocess::prepare_images(&images).unwrap();
        let individual = images
            .iter()
            .map(|image| preprocess::prepare_images(std::slice::from_ref(image)).unwrap())
            .collect();
        let merged = combine_images(individual).unwrap();
        assert_eq!(merged.pixel_values, expected.pixel_values);
        assert_eq!(merged.pixel_position_ids, expected.pixel_position_ids);
        assert_eq!(merged.num_soft_tokens, expected.num_soft_tokens);
        assert_eq!(merged.spatial_shapes, expected.spatial_shapes);
        assert_eq!(merged.max_patches, expected.max_patches);
    }
}
