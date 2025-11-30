from pathlib import Path
from openai import OpenAI

# api_key.txt 안에 실제 키 문자열만 들어있다고 가정
client = OpenAI(api_key=Path("api_key.txt").read_text().strip())

system_prompt = """
You are a robot action planner for a household robot in AI2-THOR.
Convert user commands into atomic robot actions.

Valid actions:
- GoToObject
- OpenObject
- CloseObject
- BreakObject
- SliceObject
- SwitchOn
- SwitchOff
- CleanObject
- PickupObject
- PutObject
- ThrowObject
- PullObject

Always output ONLY the JSON plan.
"""

schema = {
    "name": "robot_plan",
    "strict": False,   # JSON만 딱 나오게 하고 싶으면 True 추천
    "schema": {
        "type": "object",
        "properties": {
            "plan": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "cmd": {"type": "string"},
                        "robot": {"type": "string"},
                        "object": {"type": "string"},
                        "receptacle": {"type": "string"}
                    },
                    "required": ["cmd", "robot", "object"]
                }
            }
        },
        "required": ["plan"]
    }
}

content = """
Task: Slice the tomato
robot1 tries to slice the tomato with a knife
But robot1 cannot reach the tomato because there a Pan and a Plate blocking the way.
robot2 and robot3 are idle status. Make a new plan for robot2 and robot3 to help Robot1 finish the task.
Important: Robot needs to get close to the object to perform actions like SliceObject, PickupObject, OpenDoor.
Important: only use the valid actions listed in the system prompt.
Important: only make a plan using robot2 and robot3.
"""

message = [
    {
        "role": "system",
        "content": system_prompt
    },
    # {
    #     "role": "user",
    #     "content": "Robot1 tries to put an apple in the fridge, but the fridge is blocked by a pan. Robot2 and Robot3 are idle status. Make a new plan for Robot2 and Robot3 to help Robot1 put the apple in the fridge."
    # }
    {
        "role": "user",
        "content": content
    }
]

res = client.chat.completions.create(
    model="gpt-5.1",
    #instructions=system_prompt,
    messages=message,
    response_format={
        "type": "json_schema",
        "json_schema": schema,
    },
)

# json_schema를 쓰면 parsed 필드가 생김
print(res.choices[0].message.content)
#print("RAW:", content.text)      # 그냥 텍스트(JSON string)
#print("PARSED:", content.parsed) # 이미 파싱된 Python dict
