
select * from information_schema."tables" t

select concat('truncate table ', table_name) from information_schema.tables where table_schema = 'public'

truncate table alembic_version
truncate table study_user_attributes
truncate table trials
truncate table trial_params
truncate table trial_system_attributes
truncate table trial_user_attributes
truncate table trial_values
truncate table version_info
truncate table studies
truncate table study_system_attributes

DO $$
DECLARE
  tables CURSOR FOR
    SELECT tablename
    FROM pg_catalog.pg_tables
    WHERE schemaname = 'public';
BEGIN
  FOR t IN tables LOOP
  EXECUTE 'TRUNCATE TABLE ' || quote_ident(t.tablename) || ' CASCADE;';
  END LOOP;
END;
$$