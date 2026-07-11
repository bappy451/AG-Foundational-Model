import json

def extract_original_code():
    with open(r'C:\Users\mza0288\.gemini\antigravity\brain\92729d35-9d78-4e02-b713-bb564bf2b00e\.system_generated\logs\transcript.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if data.get('type') == 'PLANNER_RESPONSE' and 'tool_calls' in data:
                for call in data['tool_calls']:
                    if call.get('name') == 'default_api:multi_replace_file_content':
                        if 'spark_runner.py' in call['arguments'].get('TargetFile', ''):
                            print("FOUND FIRST MODIFICATION!")
                            for chunk in call['arguments']['ReplacementChunks']:
                                print("TARGET CONTENT THAT WAS REPLACED:")
                                print(chunk['TargetContent'])
                            return

if __name__ == "__main__":
    extract_original_code()
