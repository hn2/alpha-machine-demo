select s1.study_name, tv1.value, tp1.trial_id, tp1.param_name, tp1.param_value, tp1.distribution_json
from trial_params tp1, trial_values tv1, trials t1, studies s1 
where tp1.trial_id = tv1.trial_id
and tv1.trial_id = t1.trial_id
and tv1.value = (select max(value) from trial_values tv2 where tv2.trial_id = tv1.trial_id) 
and s1.study_id = t1.study_id
and s1.study_name = 'fx-ppo-10-12-2021'
