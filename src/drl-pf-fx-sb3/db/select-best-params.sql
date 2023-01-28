select tp.param_name, tp.param_value, tp.distribution_json, (select max(value) from trials t where t.trial_id = tp.trial_id) as reward
from trial_params tp
where trial_id =(select trial_id 
				from trials t 
				where value = (select max(value)
								from trials
								where study_id = (select study_id 	
												from studies s1 
												where study_name = 'fx-7-oanda-day'
												and study_id = (select max(study_id) from studies s2 where s1.study_id = s2.study_id))))
												
select tp.param_name, tp.param_value, tp.distribution_json, (select max(value) from trials t where t.trial_id = tp.trial_id) as reward
from trial_params tp
where trial_id =(select trial_id 
				from trials t 
				where value = (select max(value)
								from trials
								where study_id = (select study_id 	
												from studies s1 
												where study_name = 'fx-7-oanda-hour'
												and study_id = (select max(study_id) from studies s2 where s1.study_id = s2.study_id))))

												
select tp.param_name, tp.param_value, tp.distribution_json, (select max(value) from trials t where t.trial_id = tp.trial_id) as reward
from trial_params tp
where trial_id =(select trial_id 
				from trials t 
				where value = (select max(value)
								from trials
								where study_id = (select study_id 	
												from studies s1 
												where study_name = 'fx-7-fxcm-day'
												and study_id = (select max(study_id) from studies s2 where s1.study_id = s2.study_id))))
												
select tp.param_name, tp.param_value, tp.distribution_json, (select max(value) from trials t where t.trial_id = tp.trial_id) as reward
from trial_params tp
where trial_id =(select trial_id 
				from trials t 
				where value = (select max(value)
								from trials
								where study_id = (select study_id 	
												from studies s1 
												where study_name = 'fx-7-fxcm-hour'
												and study_id = (select max(study_id) from studies s2 where s1.study_id = s2.study_id))))