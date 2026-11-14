from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import json
import os

from cognition_base.cognition_base_tools import create_experiment, search_experiments, list_experiments
from cognition_base.database import SessionLocal
from cognition_base.schemas import ExperimentBase
from models.BaseModel import BaseModel

from utils import num_params

import json

from run.train import run

from run.config import Config

load_dotenv()


def _env_bool(name: str) -> bool | None:
    value = os.getenv(name)
    if value is None:
        return None
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _build_chat_model(*, model_name: str | None, base_url: str | None, api_key: str | None) -> ChatOpenAI:
    """Build a chat model with optional reasoning/thinking controls.

    REASONING_EFFORT: optional; passed through for OpenAI-style reasoning models.
    ENABLE_THINKING: optional; passed via chat_template_kwargs for local servers
        whose chat template supports it (e.g. llama.cpp / vLLM style servers).
    """
    kwargs = {
        "model": model_name,
        "base_url": base_url,
        "api_key": api_key,
        "temperature": 0.8,
        "timeout": 120,
        "max_retries": 3,
        # max_tokens is the widest-compatibility name for OpenAI-compatible
        # servers (including llama.cpp).
        "max_tokens": Config.MAX_COMPLETION_TOKENS,
    }

    reasoning_effort = os.getenv("REASONING_EFFORT")
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort.strip().lower()

    enable_thinking = _env_bool("ENABLE_THINKING")
    if enable_thinking is not None:
        kwargs["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": enable_thinking}
        }

    return ChatOpenAI(**kwargs)


SYSTEM_PROMPT = """
You are a concise ML architecture-search assistant for 64x64 RGB EuroSAT classification (10 land-cover classes).

Use submit_experiment(name, description, spec) to store experiments. Never invent tool results. If validation fails, fix the spec and retry.

SPEC SHAPE
A spec is a JSON object with either a single "layers" list or the sections "stem", "body", and "head" (not both):
{
  "name": "optional architecture label",
  "in_channels": 3,
  "image_size": 64,
  "num_classes": 10,
  "stem": [ ... ],
  "body": [ ... ],
  "head": [ ... ]
}
Layers run in order. Every component is an object with a "type" field. Composite components accept nested "layers" or "branches". Do not include "warmup" or "warmup_shape" in submitted specs.

AVAILABLE COMPONENTS
- identity: no options.
- sequential: {"layers": [...]}
- conv2d: in_channels?, out_channels (required), plus nn.Conv2d args such as kernel_size, stride, padding, dilation, groups, bias, padding_mode.
- conv_block: out_channels (required), in_channels?, plus Conv2d args; norm?="batch", activation?="relu", pool?=none, dropout?=0.0, bias?.
  norm: "batch" | "layer" | "group" | "instance" | "none", or {"type": "group", "num_groups": 8}.
  pool: "max" | "avg" | "adaptive_avg" | "adaptive_max" | "none", or {"type": "max", "kernel_size": 2, ...}.
  activation: relu, relu6, leaky_relu, gelu, silu, swish, mish, elu, selu, tanh, sigmoid, none.
- linear: in_features?, out_features (required), plus nn.Linear args.
- linear_block: out_features (required), in_features?, norm?=none, activation?="relu", dropout?=0.0, bias?.
- mlp: dims=[...] (required), in_features?, norm?, activation?="relu", dropout?=0.0, final_activation?=false, bias?.
- patch_embed: embed_dim or d_model (required), in_channels?, patch_size?=4, norm?=none, dropout?=0.0, bias?, plus Conv2d args except kernel_size/stride. Input (B,C,H,W) -> tokens (B,N,embed_dim).
- transformer_encoder: d_model or embed_dim (required), nhead?=8, num_layers?=1, dim_feedforward?=4*d_model, dropout?=0.0, activation?="gelu", norm_first?=false, layer_norm_eps?=1e-5, positional_encoding?="none"|"learnable"|"sinusoidal", max_tokens?=1024, output?="tokens"|"image", final_norm?. Accepts (B,N,d_model) tokens or (B,d_model,H,W) feature maps.
- residual: {"layers": [...], "shortcut": optional spec/list, "scale": 1.0}.
- parallel: {"branches": [[...], [...]], "merge": "sum"|"mean"|"max"|"mul"|"concat", "concat_dim": 1}.
- repeat: {"layers": [...], "times": N, "share_weights": false}.
- flatten: start_dim?=1, end_dim?=-1.
- reshape: shape=[...] (batch dimension excluded).
- transpose: dim0?=1, dim1?=2.
- global_pool: mode?="avg"|"max" -> (B,C).
- tokens_mean: mean over tokens -> (B,C).
- tokens_cls: index?=0 -> (B,C).
- tokens_flatten: -> (B,N*C).
- dropout: p?=0.5.
- activation: name (required) using the activation names above.
- norm: kind (batch|layer|group|instance), num_features (required), spatial?=false, plus norm kwargs.
- classifier (alias: head): num_classes (required), in_features?, hidden_dim?, activation?="relu", dropout?=0.0, bias?=true. Accepts (B,C), (B,N,C), or (B,C,H,W); reduces to (B,C) before the final Linear layers.

SHAPE RULES
- Input is always (B,3,64,64). Final output must be (B,10) logits.
- conv_block keeps 4D tensors; pooling reduces H/W; patch_embed and transformer_encoder produce 3D token tensors.
- transformer_encoder with 4D input requires channels == d_model; with 3D input requires last dim == d_model.
- End with classifier, or with global_pool/tokens_mean then linear, so the model emits 10 logits.
- Prefer explicit in_features. Omitting it creates a lazy layer initialized by BaseModel's warm-up pass.

VALID EXAMPLES
Simple CNN:
{
  "in_channels": 3, "image_size": 64, "num_classes": 10,
  "layers": [
    {"type": "conv_block", "in_channels": 3, "out_channels": 32, "kernel_size": 3, "padding": 1, "pool": "max"},
    {"type": "conv_block", "in_channels": 32, "out_channels": 64, "kernel_size": 3, "padding": 1, "pool": "max"},
    {"type": "global_pool"},
    {"type": "classifier", "num_classes": 10}
  ]
}

CNN + transformer:
{
  "in_channels": 3, "image_size": 64, "num_classes": 10,
  "stem": [
    {"type": "conv_block", "in_channels": 3, "out_channels": 32, "kernel_size": 3, "padding": 1, "pool": "max"},
    {"type": "conv_block", "in_channels": 32, "out_channels": 64, "kernel_size": 3, "padding": 1, "pool": "max"}
  ],
  "body": [
    {"type": "patch_embed", "in_channels": 64, "embed_dim": 128, "patch_size": 4},
    {"type": "transformer_encoder", "d_model": 128, "nhead": 4, "num_layers": 2, "max_tokens": 256}
  ],
  "head": [{"type": "classifier", "num_classes": 10}]
}

Return only valid JSON specs through submit_experiment. Prefer simple, trainable designs unless the user asks for something unconventional.

Constraints: maximum number of parameters must be __MAX_PARAMS__. You can run submit_experiment to validate the spec and get the number of parameters. If the number of parameters exceeds __MAX_PARAMS__, you must modify the spec to reduce the number of parameters before submitting it again.
""".replace("__MAX_PARAMS__", f"{Config.MAX_PARAMS:,}")

def extract_and_parse_json(text: str) -> dict:
    """
    Extracts and parses a JSON object from a given text string.
    Args:
        text (str): The input text containing a JSON object.
    Returns:
        dict: The parsed JSON object.
    """
    # Find the first '{' and the last '}' in the text
    start_index = text.find('{')
    end_index = text.rfind('}')

    if start_index == -1 or end_index == -1 or start_index >= end_index:
        raise ValueError("No valid JSON object found in the text.")

    # Extract the substring that is supposed to be the JSON object
    json_str = text[start_index:end_index + 1]

    try:
        # Parse the JSON string into a Python dictionary
        parsed_json = json.loads(json_str)
        return parsed_json
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}")



class BrainstormingAgent:
    def __init__(self):
        model = _build_chat_model(
            model_name=os.getenv("BRAINSTORMER_MODEL_NAME"),
            base_url=os.getenv("BRAINSTORMER_API_BASE_URL"),
            api_key=os.getenv("BRAINSTORMER_API_KEY"),
        )

        self.analysis_agent = AnalysisAgent()

        @tool
        def search_existing_experiments(query: str, on: str = "name", limit: int = 5) -> str:
            """
            Searches for experiments in the database by name, spec, or description.
            Args:
                query (str): The search query string.
                on (str): The field to search on. Must be one of 'name', 'spec', or 'description'.
            Returns:
                str: A JSON string containing a list of matching experiments.
            """
            if on not in ["name", "spec", "description"]:
                return "Invalid search field. Must be 'name', 'spec', or 'description'."

            with SessionLocal() as db:
                results = search_experiments(db, query, on, limit)

            # Convert the results to a list of dictionaries
            experiments_list = [
                {
                    "id": exp.id,
                    "name": exp.name,
                    "description": exp.description,
                    "spec": exp.spec,
                    "num_params": exp.num_params,
                    "created_at": exp.created_at.isoformat() if exp.created_at else None,
                    "train_loss": exp.train_loss,
                    "val_loss": exp.val_loss,
                    "train_acc": exp.train_acc,
                    "val_acc": exp.val_acc,
                    "train_f1": exp.train_f1,
                    "val_f1": exp.val_f1
                }
                for exp in results
            ]

            return json.dumps(experiments_list, indent=2)

        @tool
        def list_existing_experiments(limit: int = 5):
            """
            List existing experiments from the database
            """

            with SessionLocal() as db:
                results = list_experiments(db, limit)
            
            experiments_list = [
                {
                    "id": exp.id,
                    "name": exp.name,
                    "description": exp.description,
                    "spec": exp.spec,
                    "num_params": exp.num_params,
                    "created_at": exp.created_at.isoformat() if exp.created_at else None,
                    "train_loss": exp.train_loss,
                    "val_loss": exp.val_loss,
                    "train_acc": exp.train_acc,
                    "val_acc": exp.val_acc,
                    "train_f1": exp.train_f1,
                    "val_f1": exp.val_f1
                }
                for exp in results
            ]

            return json.dumps(experiments_list, indent=2)


        @tool
        def submit_experiment(name: str, description: str, spec: dict) -> str:
            """
            Submits an experiment to the database.
            Args:
                name (str): The name of the experiment.
                description (str): A brief description of the experiment.
                spec (dict): A declarative architecture spec compatible with models.BaseModel.
            """
            if not isinstance(spec, dict):
                return "Spec validation failed: 'spec' must be a JSON object."

            # Ensure the spec can be stored in the JSON database column.
            try:
                json.dumps(spec)
            except (TypeError, ValueError) as e:
                return f"Spec validation failed: spec must be JSON-serializable: {e}"

            # Build with the warm-up pass so both malformed specs and shape
            # mismatches (raised during the dummy forward) are caught here.
            try:
                model = BaseModel.from_spec({**spec, "warmup": True})
            except Exception as e:
                return f"Spec validation failed: {type(e).__name__}: {e}"

            params = num_params(model)

            if params > Config.MAX_PARAMS:
                return (
                    f"Spec validation failed: number of parameters ({params:,}) exceeds "
                    f"the maximum allowed ({Config.MAX_PARAMS:,})."
                )

            results = run(model)

            new_experiment = ExperimentBase(
                name=name,
                description=description,
                spec=spec,
                num_params=params,
                train_loss=results["train_loss"],
                val_loss=results["val_loss"],
                train_acc=results["train_acc"],
                val_acc=results["val_acc"],
                train_f1=results["train_f1"],
                val_f1=results["val_f1"]
            )

            with SessionLocal() as db:
                created_experiment = create_experiment(db, new_experiment)
            
            feedback = self.analysis_agent.call_agent(
                f"Analyze the following experiment results:\n"
                f"Name: {created_experiment.name}\n"
                f"Description: {created_experiment.description}\n"
                f"Spec: {json.dumps(created_experiment.spec, indent=2)}\n"
                f"Number of Parameters: {created_experiment.num_params}\n"
                f"Created At: {created_experiment.created_at.isoformat() if created_experiment.created_at else 'unknown'}\n"
                f"Train Loss: {created_experiment.train_loss}\n"
                f"Validation Loss: {created_experiment.val_loss}\n"
                f"Train Accuracy: {created_experiment.train_acc}\n"
                f"Validation Accuracy: {created_experiment.val_acc}\n"
                f"Train F1 Score: {created_experiment.train_f1}\n"
                f"Validation F1 Score: {created_experiment.val_f1}"
                "Return a concise summary of the experiment's performance and any recommendations for improvement inside a JSON object with keys 'summary' and 'recommendations'."
                "Example: {\"summary\": \"The experiment showed good performance with a low loss and high accuracy.\", \"recommendations\": [\"Consider adding ABC to XYZ.\"]}"
                "Do not return anything other than the JSON object, as this will be parsed by a JSON parser."
            )

            try:
                feedback_json = extract_and_parse_json(feedback[1].content)
            except ValueError as e:
                print(f"Error parsing analysis feedback: {e}")
                feedback_json = str(feedback[1].content)

            return (
                f"Experiment '{created_experiment.name}' submitted successfully.\n"
                f"Results:\n"
                f"Train Loss: {created_experiment.train_loss}\n"
                f"Validation Loss: {created_experiment.val_loss}\n"
                f"Train Accuracy: {created_experiment.train_acc}\n"
                f"Validation Accuracy: {created_experiment.val_acc}\n"
                f"Train F1 Score: {created_experiment.train_f1}\n"
                f"Validation F1 Score: {created_experiment.val_f1}\n"
                f"Analysis Feedback:"
                f"Summary: {feedback_json.get('summary', 'No summary provided.')}\n"
                f"Recommendations: {', '.join(feedback_json.get('recommendations', ['No recommendations provided.']))}"
            )
            
            

        self.agent = create_agent(
            model=model,
            tools=[submit_experiment, search_existing_experiments, list_existing_experiments],
            system_prompt=SYSTEM_PROMPT,
        )

        self.history: list[BaseMessage] = []

    def call_agent(self, message) -> str:
        history = [*self.history, HumanMessage(content=message)]

        result = self.agent.invoke({"messages": history})

        new_messages = result["messages"]
        self.history = new_messages

        return new_messages, new_messages[-1]

class AnalysisAgent:
    def __init__(self):
        model = _build_chat_model(
            model_name=os.getenv("ANALYZER_MODEL_NAME"),
            base_url=os.getenv("ANALYZER_API_BASE_URL"),
            api_key=os.getenv("ANALYZER_API_KEY"),
        )

        self.agent = create_agent(
            model=model,
            tools=[],
            system_prompt=(
                "You are a concise ML architecture-search analysis assistant. "
                "You can analyze and summarize experiments, but you cannot submit new ones."
                "Return a concise summary of the experiment's performance and any recommendations for improvement inside a JSON object with keys 'summary' and 'recommendations'."
                f"The problem set is 64x64 RGB EuroSAT classification (10 land-cover classes); the maximum number of parameters is {Config.MAX_PARAMS:,}."
            ),
        )

        self.history: list[BaseMessage] = []

    def call_agent(self, message) -> str:
        history = [*self.history, HumanMessage(content=message)]

        result = self.agent.invoke({"messages": history})

        new_messages = result["messages"]
        self.history = new_messages

        return new_messages, new_messages[-1]
