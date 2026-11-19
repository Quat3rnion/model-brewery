#[cfg(target_os = "linux")]
fn run() -> halogen_proof::Result<()> {
    use std::{ffi::OsStr, io, path::Path};
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    match args.as_slice() {
        [verb, manifest] if verb == OsStr::new("qualify") => {
            let qualified = halogen_proof::qualify(Path::new(manifest))?;
            serde_json::to_writer_pretty(io::stdout().lock(), &qualified)?;
            println!();
            Ok(())
        }
        [verb, manifest, image, timeout, consent]
            if verb == OsStr::new("prove") && consent == OsStr::new("--allow-gpu-load") =>
        {
            let qualified = halogen_proof::qualify(Path::new(manifest))?;
            let proof = halogen_proof::prove(
                &qualified,
                image.to_str().ok_or("invalid image ID")?,
                timeout.to_str().ok_or("invalid timeout")?.parse()?,
            )?;
            serde_json::to_writer_pretty(io::stdout().lock(), &proof)?;
            println!();
            Ok(())
        }
        _ => Err(concat!(
            "usage: halogen-proof qualify MANIFEST\n",
            "       halogen-proof prove MANIFEST IMAGE_ID TIMEOUT_SECONDS --allow-gpu-load",
        )
        .into()),
    }
}

#[cfg(not(target_os = "linux"))]
fn run() -> Result<(), &'static str> {
    Err("halogen-proof requires Linux")
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("halogen-proof: {err}");
            std::process::ExitCode::FAILURE
        }
    }
}
