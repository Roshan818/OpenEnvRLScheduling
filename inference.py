def parse_action(text):
   parts = text.split()

   if parts[0] == "assign_job":
       return Action(
           action_type="assign_job",
           job_id=parts[1],
           machine_id=parts[2],
       )

   return Action(action_type="wait")