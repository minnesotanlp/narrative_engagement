# narrative_engagement
umbrella project for analyzing readers' engagement in stories, including the eye tracking and highlighting annotations, and correlating them with discourse features

This repo has the following repos as subtrees:

- [Discourse feature extraction](https://github.com/minnesotanlp/emotional_story_arcs)
- [Eye tracking and statistical analysis](https://github.com/minnesotanlp/eyetracking_style)
- [Eye tracking and highlight data wrangling](https://github.com/minnesotanlp/narrative_eye_tracking_analysis)
- [Narrative transportation project from Spring '22](https://github.com/kelseyneis/narrative_transportation)

## fetching and pushing to subtree repos

### fetching

1. Add the repos as remotes

```
git remote add -f emotional_story_arcs git@github.com:minnesotanlp/emotional_story_arcs.git
git remote add -f eyetracking_style git@github.com:minnesotanlp/eyetracking_style.git
git remote add -f narrative_eye_tracking_analysis git@github.com:minnesotanlp/narrative_eye_tracking_analysis.git
```

2. Add the subtrees
  
```
git subtree add --prefix emotional_story_arcs emotional_story_arcs main --squash
git subtree add --prefix eyetracking_style eyetracking_style main --squash
git subtree add --prefix narrative_eye_tracking_analysis/ narrative_eye_tracking_analysis/main --squash
```

3. 