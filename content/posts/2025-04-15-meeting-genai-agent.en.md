---
title: Building a Multilingual Meeting Assistant with LangGraph and Generative AI
date: 2025-04-15
description: In today's fast-paced business environment, effective meeting management has become more crucial than ever. With the rise of global teams and remote work, meetings often involve participants from different regions speaking various languages. This creates a need for tools that can not only summarize discussions but also make them accessible across language barriers.
In this article, I'll walk through how I built a smart meeting assistant using LangGraph and Google's Gemini model that can:
- Extract key information from meeting transcripts
- Identify and organize action items with assignees
- Translate summaries into multiple languages
- Structure everything into a clean, organized format
draft: false
author: "Momiji"
categories: ["Technology"]
tags: ["genai", "agent", "AI", "LLM", "Gemini", "generativeai", "google", "kaggle", "langraph", "langchain"]

---


# Building a Multilingual Meeting Assistant with LangGraph and Generative AI

In today's fast-paced business environment, effective meeting management has become more crucial than ever. With the rise of global teams and remote work, meetings often involve participants from different regions speaking various languages. This creates a need for tools that can not only summarize discussions but also make them accessible across language barriers.

In this article, I'll walk through how I built a smart meeting assistant using LangGraph and Google's Gemini model that can:
- Extract key information from meeting transcripts
- Identify and organize action items with assignees
- Translate summaries into multiple languages
- Structure everything into a clean, organized format


Kaggle Notebook: [Kaggle Notebook](https://www.kaggle.com/code/hellomomiji/capstone-project-gen-ai-intensive-course-2025q1)

Github: [Github Repository](https://github.com/hellomomiji/multilingual-meeting-genai-agent)


## Gen AI Capabilities Demonstrated

This project showcases several advanced generative AI capabilities:

1. **Structured Output/JSON Mode**: The system generates consistently formatted JSON outputs with defined schemas for titles, summaries, key points, and action items.

2. **Few-Shot Prompting**: Example meeting transcripts and their corresponding structured outputs are provided to guide the model toward desired results.

3. **Agents**: LangGraph is used to create a multi-step agent system with specialized nodes for different tasks (summarization, action item extraction, translation).

4. **Function Calling**: The implementation leverages tool functions to encapsulate specific capabilities within the agent workflow.

5. **Document Understanding**: The system analyzes and extracts meaningful information from meeting transcripts, including identifying speakers, tasks, and deadlines.

## The Problem: Information Loss in Multilingual Meetings

Consider this scenario: A global executive team with members from the US, China, and Japan meets to discuss their Q2 strategy. The VP from China reports warehousing delays in Eastern China, while the Japan team mentions shipping delays and user support challenges. Even with live interpretation during the meeting, extracting actionable insights efficiently from the transcript later requires significant manual effort.

Current solutions often focus on just transcription or basic summarization, missing the crucial step of organizing action items and making content accessible across languages.

## The Solution: A Structured Meeting Processing Agent

The solution is an intelligent meeting assistant that leverages generative AI to transform raw meeting transcripts into structured, actionable information. The system I built has three main components:

1. **Meeting Summarization**: Extracts the title, summary, and key points
2. **Action Item Extraction**: Identifies tasks, assignees, and deadlines
3. **Multilingual Translation**: Translates the outputs into requested languages

### The Architecture: LangGraph for Orchestration

This project leverages LangGraph, a framework built on top of LangChain that enables the creation of stateful, multi-step AI workflows. The architecture follows a directed graph pattern:

```python
# Define Graph Nodes
workflow = StateGraph(MeetingState)
    
# Add nodes
workflow.add_node("generate_summary", generate_meeting_summary)
workflow.add_node("extract_action_items", extract_meeting_action_items)
workflow.add_node("translate_content", translate_meeting_content)

# Define the flow
workflow.add_edge(START, "generate_summary")
workflow.add_edge("generate_summary", "extract_action_items")
workflow.add_conditional_edges(
    "extract_action_items",
    should_translate,
    {
        "translate": "translate_content",
        "end": END
    }
)
```

![graph](/images/meeting-genai-agent/graph.png)

This structure allows for a clean separation of concerns, with each node handling a specific task.

## The Implementation

### 1. Defining Structured Output

First, I defined a clear schema for the output using JSON, demonstrating structured output capabilities:

```python
meeting_schema = {
  "type": "object",
  "properties": {
    "title": {
      "type": "string",
      "description": "The title or topic of the meeting"
    },
    "summary": {
      "type": "string",
      "description": "A concise summary of the meeting discussion (1-3 sentences)"
    },
    "key_points": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "List of 3-5 important points discussed in the meeting"
    },
    "action_items": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "task": {
            "type": "string",
            "description": "Description of the task to be completed"
          },
          "assignee": {
            "type": "string",
            "description": "Name of the person assigned to the task"
          },
          "due_date": {
            "type": "string",
            "description": "Due date or deadline for the task (if mentioned)"
          }
        },
        "required": ["task", "assignee"]
      },
      "description": "List of tasks assigned during the meeting"
    }
  },
  "required": ["title", "summary", "action_items"]
}
```

### 2. Leveraging Few-Shot Learning

The quality of generated summaries and extracted action items largely depends on the model's understanding of what we want. To improve this, I implemented few-shot learning by providing example meetings and their expected structured outputs:

```python
few_shot_examples = [
    {
        "transcript": 
            """
            John: Good morning everyone, let's get started with our weekly product update. 
            Sarah: Great, I finished the frontend design for the user dashboard.
            John: Excellent work, Sarah. Tom, what's the status on the API integration?
            ...
            """,
        "output": {
            "title": "Weekly Product Update Meeting",
            "summary": "The team discussed progress on the user dashboard, API integration, and documentation. Several tasks were assigned to meet the upcoming release schedule.",
            "key_points": [...],
            "action_items": [...]
        }
    },
    # More examples...
]
```

### 3. Processing Flow

The heart of the system consists of three specialized tools implemented as function calls:

1. **Summary Generator**: Extracts title, summary, and key points
2. **Action Item Extractor**: Identifies tasks, assignees, and deadlines
3. **Translation Service**: Translates content into requested languages

For example, here's how the action item extractor works:

```python
@tool(description="Extract action items with assignees and due dates from a meeting transcript.")
def extract_action_items_tool(transcript: str) -> Dict:
    parser = JsonOutputParser()
    
    formatted_prompt = extract_action_items_prompt.format(
        example1_transcript=few_shot_examples[0]["transcript"],
        example1_action_items=json.dumps({"action_items": few_shot_examples[0]["output"]["action_items"]}, indent=2),
        example2_transcript=few_shot_examples[1]["transcript"],
        example2_action_items=json.dumps({"action_items": few_shot_examples[1]["output"]["action_items"]}, indent=2),
        transcript=transcript,
    )
    result = llm.invoke(formatted_prompt)
    parsed = parser.parse(result.content.strip("```json\n").strip("```"))
    return parsed
```

## Results: Turning Transcripts into Actionable Intelligence

When tested with a sample multilingual transcript (featuring English, Chinese, and Japanese), the system successfully:

1. Identified the meeting topic: "Q2 Global Strategy Executive Meeting"
2. Generated a concise summary capturing the key discussion points
3. Extracted key points highlighting operational strains in Asia, warehousing delays in China, shipping delays in Japan, and staffing needs
4. Identified specific action items with assignees and deadlines
5. Translated all content into requested languages while preserving the original formatting and structure

Here's a snippet of the output:

```
MEETING TITLE: Q2 Global Strategy Executive Meeting

SUMMARY:
The Q2 Global Strategy Executive Meeting addressed operational strains in Asia, particularly warehousing delays and system integration issues in China, as well as shipping delays and user support challenges in Japan. The meeting concluded with action items assigned to each regional team, focusing on logistics improvement, partnership finalization, support staffing, and AI prototype performance.

KEY POINTS:
1. Eastern China experiencing warehousing delays and needs logistics system upgrade due to new supplier software incompatibility.
2. Marketing rollout in Southeast Asia delayed due to translation quality; localization needs improvement.
3. Japan market sees positive feedback on new product line, but faces shipping delays and insufficient user support infrastructure.
4. R&D completed AI personalization prototype, but performance issues need to be resolved before demoing to the board.
5. China team is negotiating partnerships with two major e-commerce platforms and considering a new distribution center in Shenzhen.

ACTION ITEMS:
1. Prepare a roadmap for cross-regional logistics improvement (Assigned to: US Team, Due: In two weeks)
2. Outline integration resource needs (Assigned to: China Team, Due: In two weeks)
3. Finalize e-commerce partnerships (Assigned to: China Team)
4. Submit support staffing plan (Assigned to: Japan Team, Due: In two weeks)
5. Submit performance benchmarks for the AI prototype (Assigned to: Japan Team, Due: In two weeks)
```

The system also successfully created translations in Chinese, Japanese, and other requested languages, preserving the structure and detailed information:

```
--- Chinese ---
TITLE: 第二季度全球战略执行会议
SUMMARY: 第二季度全球战略执行会议讨论了亚洲地区的运营压力，特别是中国的仓储延误和系统集成问题，以及日本的运输延误和用户支持挑战。会议结束时，为每个区域团队分配了行动项目，重点关注物流改进、合作伙伴关系最终确定、支持人员配置和人工智能原型性能。

KEY POINTS:
1. 华东地区正经历仓储延误，由于新的供应商软件不兼容，需要进行物流系统升级。
...

--- Japanese ---
TITLE: 第2四半期グローバル戦略エグゼクティブ会議
SUMMARY: 第2四半期グローバル戦略エグゼクティブ会議では、アジアにおける業務上の負担、特に中国における倉庫の遅延とシステム統合の問題、および日本における出荷の遅延とユーザーサポートの課題について議論されました。会議は、ロジスティクスの改善、パートナーシップの最終決定、サポート要員の増強、およびAIプロトタイプのパフォーマンスに焦点を当て、各地域チームに割り当てられたアクションアイテムで締めくくられました。
...
```

## Limitations and Future Improvements

While the current implementation demonstrates the power of generative AI for meeting management, several limitations exist:

1. **Error Handling**: The system could benefit from more robust error handling, especially for unusual transcript formats or ambiguous action items.

2. **Speaker Identification**: Currently, the system doesn't track who said what beyond what's mentioned in the transcript. Adding speaker diarization would improve context.

3. **Real-time Processing**: The system works with completed transcripts. A future version could process meetings in real-time, updating summaries and action items as the meeting progresses.

4. **Integration with Task Management Systems**: Automatically creating tasks in systems like Jira, Asana, or Microsoft Planner would increase productivity.

5. **Expanding Gen AI Capabilities**: Future versions could incorporate additional capabilities like:
   - **Retrieval Augmented Generation (RAG)**: Connecting meeting content with relevant company documentation
   - **Long Context Windows**: Processing extremely lengthy meetings without losing context
   - **Embeddings**: Creating semantic representations of meetings for better cross-meeting insights

## Conclusion

This project demonstrates how generative AI can transform meeting management by automating the extraction of key information, action items, and providing multilingual support. By leveraging LangGraph for orchestration and Gemini for understanding and generation, the solution creates a structured, actionable output from messy meeting transcripts.

The implementation showcases multiple advanced Gen AI capabilities—structured output, few-shot prompting, agents, function calling, and document understanding—working together to solve a real business problem.

As organizations continue to operate globally, tools like this will become increasingly valuable for breaking down language barriers and ensuring that meetings lead to concrete actions rather than just more conversations.

The combination of structured workflows through LangGraph and the powerful language capabilities of large language models points to a future where AI assistants become integral parts of our collaborative processes, helping teams work more effectively across languages and time zones.