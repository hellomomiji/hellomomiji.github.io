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

Meeting notes are often inconsistent, action items get lost, and language barriers create additional challenges. Consider this scenario: a product team with members from the US, China, and Japan meets to discuss a product launch. While the transcript captures everything, extracting actionable insights efficiently requires significant manual effort.

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

workflow.add_edge("translate_content", END)

```

The visualized directed graph:

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

1. Identified the meeting topic: "Q2 Product Launch Planning Meeting"
2. Generated a concise summary capturing the key discussion points
3. Extracted five key points highlighting the most important information
4. Identified five specific action items with assignees and deadlines where mentioned
5. Translated all content into Chinese and Japanese while preserving the original formatting and structure

Here's a snippet of the output:

```
==================================================
MEETING TITLE: Q2 Product Launch Planning Meeting
==================================================

SUMMARY:
The team discussed the progress of the Q2 product launch, including backend integration, UI mockups, and analytics platform selection. Action items were assigned for landing page content in English, Chinese, and Japanese, as well as testing Firebase and Mixpanel.

KEY POINTS:
1. Backend integration is complete and ready for final QA
2. Design team to submit final UI mockups by next week
3. Firebase and Mixpanel to be tested in staging by Thursday
4. Landing page content updates needed in English, Chinese, and Japanese
5. Final review meeting scheduled for next Monday at 10 AM Pacific

ACTION ITEMS:
1. Coordinate with marketing for English content (Assigned to: John, Due: None)
2. Handle Chinese version of the landing page content (Assigned to: Li Wei, Due: None)
3. Take care of the Japanese translation (Assigned to: Yuki, Due: None)
4. Test Firebase and Mixpanel on staging (Assigned to: Yuki & Li Wei, Due: Thursday)
5. Finalize UI mockups (Assigned to: Design team, Due: Next week)

==================================================
TRANSLATIONS
==================================================

--- Chinese ---
TITLE: 第二季度产品发布计划会议
SUMMARY: 团队讨论了第二季度产品发布的进展，包括后端集成、UI 模型以及分析平台选择。行动事项已分配，包括英文、中文和日文的着陆页内容，以及测试 Firebase 和 Mixpanel。

KEY POINTS:
1. 后端集成已完成，准备进行最终的质量保证 (QA)
2. 设计团队将于下周提交最终 UI 模型
3. Firebase 和 Mixpanel 将于周四在暂存环境中进行测试
4. 需要更新英文、中文和日文的着陆页内容
5. 最终审核会议安排在下周一太平洋时间上午 10 点

ACTION ITEMS:
1. 与市场部协调英文内容 (Assigned to: John, Due: None)
2. 处理着陆页内容的中文版本 (Assigned to: Li Wei, Due: None)
3. 负责日语翻译 (Assigned to: Yuki, Due: None)
4. 在暂存环境中测试 Firebase 和 Mixpanel (Assigned to: Yuki & Li Wei, Due: Thursday)
5. 最终确定 UI 模型 (Assigned to: Design team, Due: Next week)

--- Japanese ---
TITLE: 第2四半期製品ローンチ計画会議
SUMMARY: チームは、第2四半期の製品ローンチの進捗状況について議論しました。これには、バックエンド統合、UIモックアップ、および分析プラットフォームの選定が含まれます。ランディングページのコンテンツ（英語、中国語、日本語）、およびFirebaseとMixpanelのテストに関するアクションアイテムが割り当てられました。

KEY POINTS:
1. バックエンド統合は完了し、最終QAの準備が完了しました
2. デザインチームは、来週までに最終UIモックアップを提出します
3. FirebaseとMixpanelは、木曜日までにステージング環境でテストされます
4. ランディングページのコンテンツ更新が英語、中国語、日本語で必要です
5. 最終レビュー会議は、太平洋時間で来週月曜日の午前10時に予定されています

ACTION ITEMS:
1. 英語コンテンツについてマーケティングと連携 (Assigned to: John, Due: None)
2. ランディングページの中国語版コンテンツを担当 (Assigned to: Li Wei, Due: None)
3. 日本語訳を担当 (Assigned to: Yuki, Due: None)
4. ステージング環境でFirebaseとMixpanelをテスト (Assigned to: Yuki & Li Wei, Due: Thursday)
5. UIモックアップを最終決定 (Assigned to: Design team, Due: Next week)
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