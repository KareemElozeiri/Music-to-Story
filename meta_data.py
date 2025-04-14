class MetaData:
    mood_map = {
            "major_happy": ["joyful", "triumphant", "celebratory", "optimistic"],
            "major_calm": ["peaceful", "serene", "gentle", "nostalgic"],
            "minor_sad": ["melancholic", "sorrowful", "longing", "regretful"],
            "minor_intense": ["mysterious", "suspenseful", "dramatic", "intense"],
            "dissonant": ["chaotic", "unsettling", "confusing", "surreal"]
    }
    
    tempo_map = {
            "very_slow": ["slow-paced", "methodical", "reflective", "deliberate"],
            "slow": ["relaxed", "contemplative", "steady", "unhurried"],
            "moderate": ["balanced", "flowing", "steady", "regular"],
            "fast": ["energetic", "lively", "dynamic", "active"],
            "very_fast": ["frantic", "urgent", "racing", "exhilarating"]
    }
    
    dynamic_map =  {
            "quiet": ["subtle", "intimate", "delicate", "whispered"],
            "moderate": ["balanced", "controlled", "steady", "moderate"],
            "loud": ["powerful", "bold", "commanding", "intense"]
    }
    
    instrument_map = {
            "strings": ["flowing", "emotional", "expressive", "rich"],
            "brass": ["bold", "majestic", "assertive", "bright"],
            "woodwinds": ["playful", "whimsical", "agile", "light"],
            "percussion": ["rhythmic", "grounding", "primal", "structured"],
            "electronic": ["modern", "innovative", "synthetic", "futuristic"],
            "piano": ["reflective", "nuanced", "intimate", "dynamic"],
            "vocals": ["personal", "direct", "emotive", "human"]
    }
    
    story_templates = [
            "In a {mood} world, where time moved {tempo}, a {character} embarked on a journey that was {dynamics} in nature.",
            "The {mood} atmosphere surrounded everything as {character} moved {tempo} through the {setting}, aware of the {dynamics} presence around them.",
            "It was a {mood} day when the {dynamics} sound echoed across the {setting}, causing everyone to move {tempo} in response.",
            "{Character} felt {mood} as they {tempo} traversed the {setting}, their heart beating {dynamics} with each step.",
            "The {setting} had always been {mood}, but today it felt different as {character} {tempo} explored its {dynamics} corners."
    ]
    
    character_types = [
            "weary traveler", "curious explorer", "reluctant hero", "wise elder", 
            "innocent child", "skilled craftsperson", "determined scientist",
            "mysterious stranger", "passionate artist", "stoic warrior"
    ]
    settings = [
            "ancient forest", "bustling city", "desolate wasteland", "serene coastline",
            "underground cavern", "floating islands", "mountain peak", "endless desert",
            "futuristic metropolis", "medieval village", "starlit cosmos", "mystic realm"
    ]