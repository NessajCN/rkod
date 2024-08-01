use std::{collections::HashMap, fs};

use reqwest::{
    header::{HeaderMap, CONTENT_TYPE},
    Client,
};
use serde::{Deserialize, Serialize};
use tokio::{runtime::Builder, sync::mpsc};
use tracing::{error, info, warn};

pub type OdResults = Vec<(String, f32, [f32; 4])>;

#[derive(Serialize)]
struct OdRequest<'a> {
    device: &'a str,
    objects: HashMap<&'a str, u32>,
}

impl<'a> OdRequest<'a> {
    fn new(od_res: &'a OdResults, device: &'a str) -> Self {
        let mut objects: HashMap<&str, u32> = HashMap::new();
        for r in od_res.iter() {
            objects.entry(&r.0).and_modify(|e| *e += 1).or_insert(1);
        }
        Self { device, objects }
    }
}

#[derive(Debug)]
pub enum UpError {
    NoToken,
    Unauthorized,
    PostFailed(String),
    ReqwestError(String),
    ChannelError(String),
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
        let c = fs::read_to_string("config.toml").expect("Error reading config.toml");
        toml::from_str::<Config>(&c).expect("Error parsing config.toml")
    }
}

#[derive(Serialize)]
struct TokenReqPayload<'a> {
    name: &'a str,
    password: &'a str,
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

    async fn get_token(&mut self) -> Result<String, UpError> {
        let token_req = TokenReqPayload {
            name: &self.config.username,
            password: &self.config.passwd,
        };
        let api_url = format!("{}getToken", &self.config.api_prefix);
        let res = self.client.post(api_url).json(&token_req).send().await?;

        match res.status().as_u16() {
            200u16 => {
                let token_res = res.json::<TokenRes>().await?;
                self.token = Some(token_res.token.clone());
                Ok(token_res.token)
            }
            _ => {
                let msg = res.json::<TokenRes>().await?.message;
                error!("Get token request failed: {msg}");
                self.token = None;
                Err(UpError::Unauthorized)
            }
        }
    }

    async fn upload(&self, od_res: &OdResults) -> Result<(), UpError> {
        match self.token.as_ref() {
            Some(t) => {
                let mut headers = HeaderMap::new();

                headers.insert(CONTENT_TYPE, "application/json".parse().unwrap());
                headers.insert("token", t.parse().unwrap());

                let od_req = OdRequest::new(od_res, &self.config.device_name);
                let res = self
                    .client
                    .post(format!("{}od", &self.config.api_prefix))
                    .json(&od_req)
                    .headers(headers)
                    .send()
                    .await?;
                match res.status().as_u16() {
                    200u16 => Ok(()),
                    401 => Err(UpError::Unauthorized),
                    s => {
                        let response = res.json::<TokenRes>().await?;
                        warn!(
                            "upload od result response: {s} - success: {}, message: {}",
                            response.success, response.message
                        );
                        Err(UpError::PostFailed(response.message))
                    }
                }
            }
            None => Err(UpError::NoToken),
        }
    }
}

pub struct UploaderWorker {
    tx_odres: mpsc::Sender<OdResults>,
}

impl UploaderWorker {
    pub fn new() -> Self {
        let (tx_odres, mut rx_odres) = mpsc::channel(16);
        let rt = Builder::new_current_thread().enable_all().build().unwrap();
        std::thread::spawn(move || {
            rt.block_on(async move {
                let mut uploader = OdResultUploader::new();
                if let Ok(token) = uploader.get_token().await {
                    info!("token retrieved: {token}");
                }
                while let Some(res) = rx_odres.recv().await {
                    if let None = uploader.token.as_ref() {
                        if let Ok(token) = uploader.get_token().await {
                            info!("token retrieved: {token}");
                        }
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
    pub fn upload_odres(&self, od_res: OdResults) -> Result<(), UpError> {
        // Uploading threahold can be modified below in (..5, _) arm.
        match (od_res.len(), self.tx_odres.blocking_send(od_res)) {
            (..5, _) => Ok(()),
            (_, Ok(_)) => Ok(()),
            (_, Err(e)) => Err(UpError::ChannelError(e.to_string())),
        }
    }
}
