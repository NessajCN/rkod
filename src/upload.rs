use std::{collections::HashMap, fs, sync::Arc};

use chrono::{DateTime, Utc};
use reqwest::{
    header::{HeaderMap, CONTENT_TYPE},
    Client,
};
use serde::{Deserialize, Serialize};
use tokio::{
    runtime::Builder,
    sync::{mpsc, Mutex},
};
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

    async fn upload(
        &self,
        od_res: &OdResults,
        tick: Arc<Mutex<DateTime<Utc>>>,
    ) -> Result<(), UpError> {
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
                    200u16 => {
                        let mut t = tick.lock().await;
                        *t = Utc::now();
                        Ok(())
                    }
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
        let (tx_odres, mut rx_odres) = mpsc::channel::<OdResults>(16);
        let rt = Builder::new_current_thread().enable_all().build().unwrap();
        std::thread::spawn(move || {
            rt.block_on(async move {
                let mut uploader = OdResultUploader::new();
                if let Ok(token) = uploader.get_token().await {
                    info!("token retrieved: {token}");
                }
                let tick = Arc::new(Mutex::new(Utc::now()));
                while let Some(res) = rx_odres.recv().await {
                    let mut last_tick = tick.lock().await;
                    let delta = Utc::now() - *last_tick;

                    // Upload results every 10 seconds unless staff crowded 5 more
                    // or unhelmed individual detected
                    if delta.num_seconds() < 10
                        && !res.iter().any(|r| r.0 == "person")
                        && res.len() < 5
                    {
                        continue;
                    }

                    if let None = uploader.token.as_ref() {
                        if let Ok(token) = uploader.get_token().await {
                            info!("token retrieved: {token}");
                        } else {
                            warn!("failed to retrieve token");
                            *last_tick = Utc::now();
                            continue;
                        }
                    }
                    // drop last_tick to unlock.
                    drop(last_tick);

                    if let Err(UpError::Unauthorized) = uploader.upload(&res, tick.clone()).await {
                        uploader.token = None;
                    }
                }
            })
        });
        Self { tx_odres }
    }
    pub fn upload_odres(&self, od_res: OdResults) -> Result<(), UpError> {
        // Upload results only if including `person` or 5 more `hat`.
        // if od_res.iter().any(|res| res.0 == "person") || od_res.len() > 5 {
        //     match self.tx_odres.blocking_send(od_res) {
        //         Ok(_) => Ok(()),
        //         Err(e) => Err(UpError::ChannelError(e.to_string())),
        //     }
        // } else {
        //     Ok(())
        // }

        match self.tx_odres.blocking_send(od_res) {
            Ok(_) => Ok(()),
            Err(e) => Err(UpError::ChannelError(e.to_string())),
        }
    }
}
