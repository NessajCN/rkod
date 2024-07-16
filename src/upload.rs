use std::{
    fs, io,
    sync::{Arc, Mutex},
};

use reqwest::{
    header::{HeaderMap, CONTENT_TYPE},
    Client,
};
use serde::{Deserialize, Serialize};
use tokio::{runtime::Builder, sync::mpsc};
use tracing::{error, warn};

pub type OdResults = Vec<(String, f32, [f32; 4])>;

enum UpError {
    NoToken,
    Unauthorized,
    PostFailed(String),
    ReqwestError(String),
}

impl From<reqwest::Error> for UpError {
    fn from(value: reqwest::Error) -> Self {
        let val = format!("{value}");
        Self::ReqwestError(val)
    }
}

#[derive(Clone, Serialize, Debug, Deserialize)]
struct Config {
    #[serde(rename = "device")]
    device_name: String,
    #[serde(rename = "name")]
    username: String,
    #[serde(rename = "pwd")]
    passwd: String,
    #[serde(rename = "postUrlPrefix")]
    api_prefix: String,
}

impl Config {
    pub fn new() -> Self {
        let c = fs::read_to_string("../rtcrs/config.toml").expect("Error reading config.toml");
        toml::from_str::<Config>(&c).expect("Error parsing config.toml")
    }
}

#[derive(Serialize)]
struct TokenReqPayload {
    name: String,
    password: String,
}

impl TokenReqPayload {
    fn new(name: &str, password: &str) -> Self {
        Self {
            name: name.into(),
            password: password.into(),
        }
    }
}

#[derive(Deserialize)]
struct TokenRes {
    message: String,
    success: bool,
    token: String,
}

#[derive(Clone)]
struct OdResultUploader {
    config: Config,
    client: Client,
    token: Option<String>,
}

impl OdResultUploader {
    fn new() -> Self {
        let client = Client::new();
        let config = Config::new();
        let token = None;

        OdResultUploader {
            config,
            client,
            token,
        }
    }

    async fn get_token(&mut self) -> Result<(), UpError> {
        let token_req = TokenReqPayload::new(&self.config.username, &self.config.passwd);
        let api_url = format!("{}getToken", &self.config.api_prefix);
        let res = self.client.post(api_url).json(&token_req).send().await?;

        match res.status().as_u16() {
            200u16 => {
                let token_res = res.json::<TokenRes>().await?;
                self.token = Some(token_res.token);
            }
            _ => {
                let msg = res.json::<TokenRes>().await?.message;
                error!("Get token request failed: {msg}");
                self.token = None;
                return Err(UpError::Unauthorized);
            }
        }
        Ok(())
    }

    async fn upload(&self, od_res: &OdResults) -> Result<(), UpError> {
        match self.token.as_ref() {
            Some(t) => {
                let mut headers = HeaderMap::new();

                headers.insert(CONTENT_TYPE, "application/json".parse().unwrap());
                headers.insert("token", t.parse().unwrap());

                let res = self
                    .client
                    .post(format!("{}od", &self.config.api_prefix))
                    .json(od_res)
                    .headers(headers.to_owned())
                    .send()
                    .await?;
                let ret = match res.status().as_u16() {
                    200u16 => Ok(()),
                    401 => Err(UpError::Unauthorized),
                    s => {
                        let msg = res.json::<TokenRes>().await?.message;
                        warn!("upload od result response: {s} - {msg}");
                        Err(UpError::PostFailed(msg))
                    }
                };
                return ret;
            }
            None => {
                return Err(UpError::NoToken);
            }
        }
    }
}

struct UploaderWorker {
    tx_odres: mpsc::Sender<OdResults>,
}

impl UploaderWorker {
    fn new() -> Self {
        let (tx_odres, mut rx_odres) = mpsc::channel(16);
        let rt = Builder::new_current_thread().enable_all().build().unwrap();
        std::thread::spawn(move || {
            rt.block_on(async move {
                let mut uploader = OdResultUploader::new();
                let _ = uploader.get_token();
                while let Some(res) = rx_odres.recv().await {
                    if let None = uploader.token.as_ref() {
                        let _ = uploader.get_token();
                    }
                    let up = uploader.clone();
                    tokio::spawn(async move {
                        let _ = up.upload(&res).await;
                    });
                }
            })
        });
        Self { tx_odres }
    }
}
