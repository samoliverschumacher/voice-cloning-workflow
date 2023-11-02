import sys
from pathlib import Path

scripts_dir = Path(__file__).parent.parent / "scripts"
print(scripts_dir)
sys.path.append(str(scripts_dir))

from process_transcript import _process_content


input_text=("And so I looked around and I looked at what psychology seemed to do pretty well; "
     "it did misery and suffering pretty well. And indeed when I was on an aeroplane and"
     "I introduced myself to my seatmate and they'd ask me what I did and I told them, "
     "they'd move away from me. Now, by the way, when I introduce myself to my seatmate "
     "they move toward me, when I say what I work on is happiness and positive psychology. "
     "That's a very interesting change.And it occurred to me that what psychology did (misery"
     "and suffering), did pretty well, but I have to say that clinical psychology as I know it"
     "has reached the following dead end. I have written five editions of Abnormal Psychology over "
     "the last 30 years, and every five years I review the literature on what has gotten better "
     "in drugs and psychotherapy. And the answer is nothing. Except for one disorder and one"
     "development, we are in the same place we were 30 years ago. And one, by the way, is"
     "Viagra. So Viagra actually has changed…it's the only form of drugs or psychotherapy that has "
     "made a significant inroad that has changed. But basically, to summarise that literature, we've "
     "reached a 60% barrier. The best we can do when we are trying to alleviate misery is about 60% "
     "effectiveness against the 40% placebo rate, and that really hasn't changed.So as president of "
     "the American Psychological Association, I said, look, when you lie in bed at night you are "
     "generally not thinking about how to go from -8 to -5, you're thinking about how to go from +3"
     "to +6 in life. Psychologists have never worked on this, we've never worked on happiness, "
     "well-being, the stuff that is above zero. So my call to my fellow psychologists around the world was "
     "let's work on happiness, let's ask the same questions that we asked about the alleviation of misery about"
     "the building of well-being. So that was my theme. And then it became my mission, and I've spent most of "
     "the last 20 years working on this.And the first question was try to define happiness. Good heavens, what "
     "a difficult word that was and how useless it is if you see a patient or a friend to say, 'Be happy.' So what I tried to do was say what are the components of well-being that we can actually work on? And the acronym for that is PERMA. I'll take you through this. I think there are five elements that free people who aren't suffering choose to pursue.The first is pleasant emotions, like happiness, contentment, joy, rapture. The second is flow, being completely engaged. So, looking around at you, it looks to me like maybe 80% of you are completely engaged with what I'm saying. And we know now that the other 20% of you are having sexual fantasies. But it turns out the second thing that people pursue is flow, when time stops for you. The third is people pursue positive relationships, they pursue relationships, we are hive creatures, we are like termites and bees and wasps, we are social hive creatures.The fourth thing that we pursue ineluctably is being part of something larger than we are, meaning and purpose, belonging to and serving something bigger than the self. It turns out…I'm not preaching, it turns out we are made that way. Interestingly, about the last two, if you are depressed now, and about 15% of Australians are depressed right now, I'm often asked what's the one thing you should go out and do that is most likely to give you at least temporary relief? And the answer is to leave this lecture, go out and find someone who needs help and help them. Turns out we are wired to help other people.And the final thing that people pursue is competence, achievement, accomplishment. So that's what PERMA is about. So that's the past, has been an attempt to define what happiness is or well-being is, an attempt to build interventions that build well-being, not interventions that alleviate suffering necessarily but interventions that make people happier.")
expected = "AND|SO|I|LOOKED|AROUND|AND|I|LOOKED|AT|WHAT|PSYCHOLOGY|SEEMED|TO|DO|PRETTY|WELL|IT|DID|MISERY|AND|SUFFERING|PRETTY|WELL|AND|INDEED|WHEN|I|WAS|ON|AN|AEROPLANE|AND|I|INTRODUCED|MYSELF|TO|MY|SEATMATE|AND|THEYD|ASK|ME|WHAT|I|DID|AND|I|TOLD|THEM|THEYD|MOVE|AWAY|FROM|ME|NOW|BY|THE|WAY|WHEN|I|INTRODUCE|MYSELF|TO|MY|SEATMATE|THEY|MOVE|TOWARD|ME|WHEN|I|SAY|WHAT|I|WORK|ON|IS|HAPPINESS|AND|POSITIVE|PSYCHOLOGY|THATS|A|VERY|INTERESTING|CHANGE|AND|IT|OCCURRED|TO|ME|THAT|WHAT|PSYCHOLOGY|DID|MISERY|AND|SUFFERING|DID|PRETTY|WELL|BUT|I|HAVE|TO|SAY|THAT|CLINICAL|PSYCHOLOGY|AS|I|KNOW|IT|HAS|REACHED|THE|FOLLOWING|DEAD|END|I|HAVE|WRITTEN|FIVE|EDITIONS|OF|ABNORMAL|PSYCHOLOGY|OVER|THE|LAST|THIRTY|YEARS|AND|EVERY|FIVE|YEARS|I|REVIEW|THE|LITERATURE|ON|WHAT|HAS|GOTTEN|BETTER|IN|DRUGS|AND|PSYCHOTHERAPY|AND|THE|ANSWER|IS|NOTHING|EXCEPT|FOR|ONE|DISORDER|AND|ONE|DEVELOPMENT|WE|ARE|IN|THE|SAME|PLACE|WE|WERE|THIRTY|YEARS|AGO|AND|ONE|BY|THE|WAY|IS|VIAGRA|SO|VIAGRA|ACTUALLY|HAS|CHANGED|ITS|THE|ONLY|FORM|OF|DRUGS|OR|PSYCHOTHERAPY|THAT|HAS|MADE|A|SIGNIFICANT|INROAD|THAT|HAS|CHANGED|BUT|BASICALLY|TO|SUMMARISE|THAT|LITERATURE|WEVE|REACHED|A|SIXTY|PERCENT|BARRIER|THE|BEST|WE|CAN|DO|WHEN|WE|ARE|TRYING|TO|ALLEVIATE|MISERY|IS|ABOUT|SIXTY|PERCENT|EFFECTIVENESS|AGAINST|THE|FORTY|PERCENT|PLACEBO|RATE|AND|THAT|REALLY|HASNT|CHANGED|SO|AS|PRESIDENT|OF|THE|AMERICAN|PSYCHOLOGICAL|ASSOCIATION|I|SAID|LOOK|WHEN|YOU|LIE|IN|BED|AT|NIGHT|YOU|ARE|GENERALLY|NOT|THINKING|ABOUT|HOW|TO|GO|FROM|MINUS|EIGHT|TO|MINUS|FIVE|YOURE|THINKING|ABOUT|HOW|TO|GO|FROM|PLUS|THREE|TO|PLUS|SIX|IN|LIFE|PSYCHOLOGISTS|HAVE|NEVER|WORKED|ON|THIS|WEVE|NEVER|WORKED|ON|HAPPINESS|WELL|BEING|THE|STUFF|THAT|IS|ABOVE|ZERO|SO|MY|CALL|TO|MY|FELLOW|PSYCHOLOGISTS|AROUND|THE|WORLD|WAS|LETS|WORK|ON|HAPPINESS|LETS|ASK|THE|SAME|QUESTIONS|THAT|WE|ASKED|ABOUT|THE|ALLEVIATION|OF|MISERY|ABOUT|THE|BUILDING|OF|WELL|BEING|SO|THAT|WAS|MY|THEME|AND|THEN|IT|BECAME|MY|MISSION|AND|IVE|SPENT|MOST|OF|THE|LAST|TWENTY|YEARS|WORKING|ON|THIS|AND|THE|FIRST|QUESTION|WAS|TRY|TO|DEFINE|HAPPINESS|GOOD|HEAVENS|WHAT|A|DIFFICULT|WORD|THAT|WAS|AND|HOW|USELESS|IT|IS|IF|YOU|SEE|A|PATIENT|OR|A|FRIEND|TO|SAY|BE|HAPPY|SO|WHAT|I|TRIED|TO|DO|WAS|SAY|WHAT|ARE|THE|COMPONENTS|OF|WELL|BEING|THAT|WE|CAN|ACTUALLY|WORK|ON|AND|THE|ACRONYM|FOR|THAT|IS|PERMA|ILL|TAKE|YOU|THROUGH|THIS|I|THINK|THERE|ARE|FIVE|ELEMENTS|THAT|FREE|PEOPLE|WHO|ARENT|SUFFERING|CHOOSE|TO|PURSUE|THE|FIRST|IS|PLEASANT|EMOTIONS|LIKE|HAPPINESS|CONTENTMENT|JOY|RAPTURE|THE|SECOND|IS|FLOW|BEING|COMPLETELY|ENGAGED|SO|LOOKING|AROUND|AT|YOU|IT|LOOKS|TO|ME|LIKE|MAYBE|EIGHTY|PERCENT|OF|YOU|ARE|COMPLETELY|ENGAGED|WITH|WHAT|IM|SAYING|AND|WE|KNOW|NOW|THAT|THE|OTHER|TWENTY|PERCENT|OF|YOU|ARE|HAVING|SEXUAL|FANTASIES|BUT|IT|TURNS|OUT|THE|SECOND|THING|THAT|PEOPLE|PURSUE|IS|FLOW|WHEN|TIME|STOPS|FOR|YOU|THE|THIRD|IS|PEOPLE|PURSUE|POSITIVE|RELATIONSHIPS|THEY|PURSUE|RELATIONSHIPS|WE|ARE|HIVE|CREATURES|WE|ARE|LIKE|TERMITES|AND|BEES|AND|WASPS|WE|ARE|SOCIAL|HIVE|CREATURES|THE|FOURTH|THING|THAT|WE|PURSUE|INELUCTABLY|IS|BEING|PART|OF|SOMETHING|LARGER|THAN|WE|ARE|MEANING|AND|PURPOSE|BELONGING|TO|AND|SERVING|SOMETHING|BIGGER|THAN|THE|SELF|IT|TURNS|OUT|IM|NOT|PREACHING|IT|TURNS|OUT|WE|ARE|MADE|THAT|WAY|INTERESTINGLY|ABOUT|THE|LAST|TWO|IF|YOU|ARE|DEPRESSED|NOW|AND|ABOUT|FIFTEEN|PERCENT|OF|AUSTRALIANS|ARE|DEPRESSED|RIGHT|NOW|IM|OFTEN|ASKED|WHATS|THE|ONE|THING|YOU|SHOULD|GO|OUT|AND|DO|THAT|IS|MOST|LIKELY|TO|GIVE|YOU|AT|LEAST|TEMPORARY|RELIEF|AND|THE|ANSWER|IS|TO|LEAVE|THIS|LECTURE|GO|OUT|AND|FIND|SOMEONE|WHO|NEEDS|HELP|AND|HELP|THEM|TURNS|OUT|WE|ARE|WIRED|TO|HELP|OTHER|PEOPLE|AND|THE|FINAL|THING|THAT|PEOPLE|PURSUE|IS|COMPETENCE|ACHIEVEMENT|ACCOMPLISHMENT|SO|THATS|WHAT|PERMA|IS|ABOUT|SO|THATS|THE|PAST|HAS|BEEN|AN|ATTEMPT|TO|DEFINE|WHAT|HAPPINESS|IS|OR|WELL|BEING|IS|AN|ATTEMPT|TO|BUILD|INTERVENTIONS|THAT|BUILD|WELL|BEING|NOT|INTERVENTIONS|THAT|ALLEVIATE|SUFFERING|NECESSARILY|BUT|INTERVENTIONS|THAT|MAKE|PEOPLE|HAPPIER"

ans = _process_content(input_text, verbose=True)