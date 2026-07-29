from typing import Any, Dict, List
from autogen import ConversableAgent
import sys
import os
import math
import re
from pathlib import Path


def is_termination_message(message: Dict[str, Any]) -> bool:
    """
    仅当文本消息以 TERMINATE 结尾时结束当前对话。
    """
    content = message.get("content", "")
    return (
        isinstance(content, str)
        and content.rstrip().endswith("TERMINATE")
    )


def summary_without_terminate(
    sender: ConversableAgent,
    recipient: ConversableAgent,
    summary_args: Dict[str, Any],
) -> str:
    """
    保留阶段的最后一条文本消息，但移除其末尾的 TERMINATE。
    """
    last_message = recipient.last_message(sender)

    if not last_message:
        return ""

    content = last_message.get("content", "")

    if not isinstance(content, str):
        return ""

    return re.sub(
        r"\s*TERMINATE\s*$",
        "",
        content,
        flags=re.IGNORECASE,
    ).strip()


def normalize_restaurant_name(name: str) -> str:
    """
    统一餐厅名称格式。

    例如：
    Example Bistro、Example-Bistro、example bistro
    都会转换为 examplebistro。
    """
    return re.sub(r"[^a-z0-9]", "", name.casefold())


def fetch_restaurant_data(
    restaurant_name: str
) -> Dict[str, List[str]]:
    """
    从 restaurant-data.txt 中读取指定餐厅的全部评论。

    返回格式：
    {
        "餐厅名称": ["评论1", "评论2", ...]
    }
    """

    # 找到与 main.py 位于同一个文件夹中的数据文件
    data_file = Path(__file__).with_name("restaurant-data.txt")

    target_name = normalize_restaurant_name(restaurant_name)

    reviews: List[str] = []

    # 默认使用用户传入的名称。
    # 找到匹配项后，替换为数据文件中的标准名称。
    matched_restaurant_name = restaurant_name.strip()

    with data_file.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()

            # 跳过空行
            if not line:
                continue

            # 只按照第一个英文句号进行分割
            current_name, separator, review = line.partition(".")

            # 如果这一行没有句号，说明格式不正确，直接跳过
            if not separator:
                continue

            normalized_current_name = normalize_restaurant_name(current_name)

            if (
                normalized_current_name == target_name
                or normalized_current_name in target_name
            ):
                matched_restaurant_name = current_name.strip()
                reviews.append(review.strip())

    return {matched_restaurant_name: reviews}


SCORE_KEYWORDS = {
    1: ["awful", "horrible", "disgusting"],
    2: ["bad", "unpleasant", "offensive"],
    3: ["average", "uninspiring", "forgettable"],
    4: ["good", "enjoyable", "satisfying"],
    5: ["awesome", "incredible", "amazing"],
}


def get_review_analysis_prompt(
    restaurant_name: str,
    reviews: List[str],
) -> str:
    """
    生成 Review Analysis Agent 使用的提示词。
    """

    numbered_reviews = "\n".join(
        f"{index + 1}. {review}"
        for index, review in enumerate(reviews)
    )
    score_placeholders = ", ".join(
        "integer" for _ in reviews
    )

    return f"""
You are a restaurant review analysis agent.

Your task is to analyze every review for the restaurant "{restaurant_name}".

For each review, extract exactly two scores:

1. food_score:
   The score for the adjective describing the food.

2. customer_service_score:
   The score for the adjective describing customer service.

You must use only the following fixed mapping:

Score 1:
awful, horrible, disgusting

Score 2:
bad, unpleasant, offensive

Score 3:
average, uninspiring, forgettable

Score 4:
good, enjoyable, satisfying

Score 5:
awesome, incredible, amazing

Important rules:

- The mapping above is fixed.
- Do not use your own opinion.
- Do not infer a score from words outside the mapping.
- Analyze every review separately.
- Determine which scoring adjective describes food.
- Determine which scoring adjective describes customer service.
- Preserve the original review order.
- Return exactly one food score and one customer-service score per review.
- Return only valid JSON.
- Do not include Markdown code fences.
- Do not include explanations before or after the JSON.

The output must use exactly this structure:

{{
    "restaurant_name": "{restaurant_name}",
    "food_scores": [{score_placeholders}],
    "customer_service_scores": [{score_placeholders}]
}}

Reviews:

{numbered_reviews}
""".strip()


def validate_review_scores(
    analysis_data: Dict,
    expected_review_count: int,
) -> None:
    """
    检查 Review Analysis Agent 的结果是否有效。
    """

    required_keys = {
        "restaurant_name",
        "food_scores",
        "customer_service_scores",
    }

    if not isinstance(analysis_data, dict):
        raise ValueError(
            "Review analysis result must be a JSON object."
        )

    if not required_keys.issubset(analysis_data):
        raise ValueError(
            "Review analysis result is missing required keys."
        )

    food_scores = analysis_data["food_scores"]
    service_scores = analysis_data["customer_service_scores"]

    if not isinstance(food_scores, list):
        raise ValueError("Food scores must be a list.")

    if not isinstance(service_scores, list):
        raise ValueError(
            "Customer-service scores must be a list."
        )

    if len(food_scores) != expected_review_count:
        raise ValueError(
            "The number of food scores does not match "
            "the number of reviews."
        )

    if len(service_scores) != expected_review_count:
        raise ValueError(
            "The number of customer-service scores does not match "
            "the number of reviews."
        )

    valid_scores = {1, 2, 3, 4, 5}

    if any(
        isinstance(score, bool)
        or not isinstance(score, int)
        or score not in valid_scores
        for score in food_scores
    ):
        raise ValueError("Invalid food score detected.")

    if any(
        isinstance(score, bool)
        or not isinstance(score, int)
        or score not in valid_scores
        for score in service_scores
    ):
        raise ValueError(
            "Invalid customer-service score detected."
        )


def calculate_overall_score(
    restaurant_name: str,
    food_scores: List[int],
    customer_service_scores: List[int],
) -> Dict[str, float]:
    """
    根据每条评论的食物分和服务分计算餐厅总分。

    food_scores 和 customer_service_scores 中的元素
    都应该是 1～5 的整数。
    """

    if len(food_scores) != len(customer_service_scores):
        raise ValueError("食物评分和服务评分的数量必须相同。")

    if len(food_scores) == 0:
        raise ValueError("评分列表不能为空。")

    all_scores = food_scores + customer_service_scores

    if any(score < 1 or score > 5 for score in all_scores):
        raise ValueError("每个评分必须在 1 到 5 之间。")

    number_of_reviews = len(food_scores)

    weighted_score_sum = 0.0

    for food_score, service_score in zip(
        food_scores,
        customer_service_scores,
    ):
        review_score = math.sqrt(
            food_score ** 2 * service_score
        )
        weighted_score_sum += review_score

    overall_score = (
        weighted_score_sum
        * 10
        / (number_of_reviews * math.sqrt(125))
    )

    # 按作业要求至少保留三位小数精度
    return {
        restaurant_name: round(overall_score, 3)
    }


def print_score_result(score_result: Dict[str, float]) -> None:
    restaurant_name, overall_score = next(iter(score_result.items()))
    print(f"{restaurant_name} overall score: {overall_score:.3f}")


def get_data_fetch_agent_prompt(restaurant_query: str) -> str:
    """
    生成 Data Fetch Agent 使用的提示词。
    """
    return f"""
Identify the restaurant in the following user question and fetch its reviews:

{restaurant_query}

You must call fetch_restaurant_data with only the restaurant name. After the
tool result is returned, preserve every review and return the required JSON.
""".strip()


ENTRYPOINT_AGENT_SYSTEM_MESSAGE = """
You are the Supervisor and Orchestrator for a restaurant-scoring workflow.

You must coordinate the work in this exact order:
1. Fetch restaurant data.
2. Analyze every fetched review.
3. Calculate the overall score.

You execute Python functions requested by the specialized agents. You may
execute only registered Python functions. Never invent or alter reviews. Never
create food scores or customer-service scores yourself. Never estimate or
approximate the final score yourself. Pass the complete structured result from
each stage to the next stage. The final result must contain the restaurant name
and an overall score displayed with at least three digits after the decimal
point.
""".strip()


DATA_FETCH_AGENT_SYSTEM_MESSAGE = """
You are the Data Fetch Agent in a sequential restaurant-scoring workflow.

Your only responsibilities are:
1. Identify the restaurant name in the user's question.
2. Request exactly one call to fetch_restaurant_data(restaurant_name).
3. After receiving the function result, return the canonical restaurant name
   and every review from that result.

The fetch_restaurant_data tool is the only allowed source of restaurant data.
Do not open or read restaurant-data.txt yourself. Do not analyze reviews,
generate scores, calculate an overall score, invent reviews, edit reviews, or
omit reviews.

After the tool returns, respond with valid JSON, without Markdown or
explanatory text, in exactly this shape:
{
  "restaurant_name": "canonical name from the tool result",
  "reviews": ["review 1", "review 2"]
}
The reviews array must contain all returned reviews in their original order.
Only after fetch_restaurant_data has executed successfully and its complete
result has been received, write TERMINATE on its own line after the JSON. Never
write TERMINATE in the same response that requests the function call.
""".strip()


REVIEW_ANALYSIS_AGENT_SYSTEM_MESSAGE = """
You are the Review Analysis Agent in a sequential restaurant-scoring workflow.
Your input is structured JSON containing restaurant_name and reviews from the
Data Fetch Agent.

Analyze every review separately and extract exactly one food_score and exactly
one customer_service_score from each review. Use only this fixed mapping:

1: awful, horrible, disgusting
2: bad, unpleasant, offensive
3: average, uninspiring, forgettable
4: good, enjoyable, satisfying
5: awesome, incredible, amazing

Determine which mapped adjective describes the food and which describes
customer service. Do not use subjective judgment or words outside the mapping.
Do not call any Python function. Do not omit, duplicate, reorder, or modify any
review. The food_scores and customer_service_scores arrays must have equal
lengths, and each length must exactly equal the number of input reviews.

Respond with exactly these four lines, without Markdown or explanatory text:
restaurant_name: <the input restaurant name>
food_scores: [one integer per review]
customer_service_scores: [one integer per review]
TERMINATE
""".strip()


SCORING_AGENT_SYSTEM_MESSAGE = """
You are the Scoring Agent in a sequential restaurant-scoring workflow. Your
required input is the Review Analysis Agent's structured result containing
restaurant_name, food_scores, and customer_service_scores. It is supplied
through sequential-chat carryover.

First verify that food_scores and customer_service_scores have equal lengths.
Then request exactly one call to calculate_overall_score using the restaurant
name and both complete, unchanged score arrays. Do not inspect or reanalyze raw
reviews. Do not delete, reorder, change, or add scores. Do not manually
calculate, estimate, or approximate the overall score.

Only after calculate_overall_score has executed successfully and its result has
been received, respond with exactly this sentence:
<restaurant_name> has an overall score of <score>.

Format <score> with exactly three digits after the decimal point. Use only the
value returned by the tool. Then write TERMINATE on its own line. Never write
TERMINATE in the same response that requests the function call.
""".strip()


# Do not modify the signature of the "main" function.
def main(user_query: str):
    api_config = {
        "model": "gpt-4o-mini",
        "api_key": os.environ.get("OPENAI_API_KEY"),
    }

    base_url = os.environ.get("OPENAI_BASE_URL")
    if base_url:
        api_config["base_url"] = base_url

    llm_config = {
        "config_list": [api_config],
        "temperature": 0,
    }

    entrypoint_agent = ConversableAgent(
        name="entrypoint_agent",
        system_message=ENTRYPOINT_AGENT_SYSTEM_MESSAGE,
        llm_config=llm_config,
        human_input_mode="NEVER",
        is_termination_msg=is_termination_message,
        max_consecutive_auto_reply=4,
    )

    data_fetch_agent = ConversableAgent(
        name="data_fetch_agent",
        system_message=DATA_FETCH_AGENT_SYSTEM_MESSAGE,
        llm_config=llm_config,
        human_input_mode="NEVER",
        is_termination_msg=is_termination_message,
    )

    review_analysis_agent = ConversableAgent(
        name="review_analysis_agent",
        system_message=REVIEW_ANALYSIS_AGENT_SYSTEM_MESSAGE,
        llm_config=llm_config,
        human_input_mode="NEVER",
        is_termination_msg=is_termination_message,
    )

    scoring_agent = ConversableAgent(
        name="scoring_agent",
        system_message=SCORING_AGENT_SYSTEM_MESSAGE,
        llm_config=llm_config,
        human_input_mode="NEVER",
        is_termination_msg=is_termination_message,
    )

    data_fetch_agent.register_for_llm(
        name="fetch_restaurant_data",
        description="Fetch all reviews for a specified restaurant.",
    )(fetch_restaurant_data)
    entrypoint_agent.register_for_execution(
        name="fetch_restaurant_data"
    )(fetch_restaurant_data)

    scoring_agent.register_for_llm(
        name="calculate_overall_score",
        description=(
            "Calculate the overall restaurant score from the restaurant name, "
            "food scores, and customer service scores."
        ),
    )(calculate_overall_score)
    entrypoint_agent.register_for_execution(
        name="calculate_overall_score"
    )(calculate_overall_score)

    results = entrypoint_agent.initiate_chats(
        [
            {
                "recipient": data_fetch_agent,
                "message": f"""
Analyze the user's restaurant query:

{user_query}

Identify the restaurant name and call fetch_restaurant_data exactly once.
After the function result is returned, preserve and return the complete
restaurant name and every review exactly as returned, in the original order.

Do not evaluate, score, shorten, merge, rewrite, or omit any review. Do not
calculate the overall score.
""".strip(),
                "max_turns": 2,
                "summary_method": summary_without_terminate,
                "clear_history": True,
                "silent": True,
            },
            {
                "recipient": review_analysis_agent,
                "message": """
Use the complete restaurant name and reviews from the carryover.

Analyze every review exactly once using only the fixed keyword mapping defined
in your system message. Find the mapped adjective that describes food and the
mapped adjective that describes customer service in each review.

Return only one structured result containing restaurant_name, food_scores, and
customer_service_scores. Preserve review order. Do not omit, duplicate,
reorder, reinterpret, or modify any score.

The two score lists must have equal lengths, and each length must equal the
number of reviews. Do not call calculate_overall_score and do not calculate the
overall score.
""".strip(),
                "max_turns": 1,
                "summary_method": summary_without_terminate,
                "clear_history": True,
                "silent": True,
            },
            {
                "recipient": scoring_agent,
                "message": """
Use the structured restaurant_name, food_scores, and customer_service_scores
from the carryover.

Call calculate_overall_score exactly once with the complete, unmodified
arguments. Do not delete, add, change, or reorder any score, and do not
calculate or estimate the overall score yourself.

After receiving the function result, return a final sentence containing the
restaurant name and the function's score formatted with exactly three decimal
places.
""".strip(),
                "max_turns": 2,
                "summary_method": summary_without_terminate,
                "clear_history": True,
                "silent": True,
            },
        ]
    )

    if len(results) != 3:
        raise RuntimeError(
            "The sequential workflow did not complete all three stages."
        )

    final_answer = results[-1].summary

    if final_answer:
        final_answer = final_answer.strip()
        print(final_answer)

    return final_answer

# DO NOT modify this code below.
if __name__ == "__main__":
    assert len(sys.argv) > 1, "Please ensure you include a query for some restaurant when executing main."
    main(sys.argv[1])
