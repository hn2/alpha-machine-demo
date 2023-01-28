@echo off
cd E:\alpha-machine\models\sb3\day
aws s3 sync E:\alpha-machine\models\sb3\day  s3://my-forex/models/
PAUSE >nul





