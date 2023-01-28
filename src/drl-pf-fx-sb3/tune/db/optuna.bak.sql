PGDMP         	                 y            optuna    13.1    13.1 d    C           0    0    ENCODING    ENCODING        SET client_encoding = 'UTF8';
                      false            D           0    0 
   STDSTRINGS 
   STDSTRINGS     (   SET standard_conforming_strings = 'on';
                      false            E           0    0 
   SEARCHPATH 
   SEARCHPATH     8   SELECT pg_catalog.set_config('search_path', '', false);
                      false            F           1262    29810    optuna    DATABASE     c   CREATE DATABASE optuna WITH TEMPLATE = template0 ENCODING = 'UTF8' LOCALE = 'English_Israel.1252';
    DROP DATABASE optuna;
                postgres    false            �           1247    29812    studydirection    TYPE     ]   CREATE TYPE public.studydirection AS ENUM (
    'NOT_SET',
    'MINIMIZE',
    'MAXIMIZE'
);
 !   DROP TYPE public.studydirection;
       public          postgres    false            �           1247    29820 
   trialstate    TYPE     r   CREATE TYPE public.trialstate AS ENUM (
    'RUNNING',
    'COMPLETE',
    'PRUNED',
    'FAIL',
    'WAITING'
);
    DROP TYPE public.trialstate;
       public          postgres    false            �            1259    29997    alembic_version    TABLE     X   CREATE TABLE public.alembic_version (
    version_num character varying(32) NOT NULL
);
 #   DROP TABLE public.alembic_version;
       public         heap    postgres    false            �            1259    29833    studies    TABLE     o   CREATE TABLE public.studies (
    study_id integer NOT NULL,
    study_name character varying(512) NOT NULL
);
    DROP TABLE public.studies;
       public         heap    postgres    false            �            1259    29831    studies_study_id_seq    SEQUENCE     �   CREATE SEQUENCE public.studies_study_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 +   DROP SEQUENCE public.studies_study_id_seq;
       public          postgres    false    201            G           0    0    studies_study_id_seq    SEQUENCE OWNED BY     M   ALTER SEQUENCE public.studies_study_id_seq OWNED BY public.studies.study_id;
          public          postgres    false    200            �            1259    29851    study_directions    TABLE     �   CREATE TABLE public.study_directions (
    study_direction_id integer NOT NULL,
    direction public.studydirection NOT NULL,
    study_id integer NOT NULL,
    objective integer NOT NULL
);
 $   DROP TABLE public.study_directions;
       public         heap    postgres    false    644            �            1259    29849 '   study_directions_study_direction_id_seq    SEQUENCE     �   CREATE SEQUENCE public.study_directions_study_direction_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 >   DROP SEQUENCE public.study_directions_study_direction_id_seq;
       public          postgres    false    204            H           0    0 '   study_directions_study_direction_id_seq    SEQUENCE OWNED BY     s   ALTER SEQUENCE public.study_directions_study_direction_id_seq OWNED BY public.study_directions.study_direction_id;
          public          postgres    false    203            �            1259    29884    study_system_attributes    TABLE     �   CREATE TABLE public.study_system_attributes (
    study_system_attribute_id integer NOT NULL,
    study_id integer,
    key character varying(512),
    value_json character varying(2048)
);
 +   DROP TABLE public.study_system_attributes;
       public         heap    postgres    false            �            1259    29882 5   study_system_attributes_study_system_attribute_id_seq    SEQUENCE     �   CREATE SEQUENCE public.study_system_attributes_study_system_attribute_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 L   DROP SEQUENCE public.study_system_attributes_study_system_attribute_id_seq;
       public          postgres    false    208            I           0    0 5   study_system_attributes_study_system_attribute_id_seq    SEQUENCE OWNED BY     �   ALTER SEQUENCE public.study_system_attributes_study_system_attribute_id_seq OWNED BY public.study_system_attributes.study_system_attribute_id;
          public          postgres    false    207            �            1259    29866    study_user_attributes    TABLE     �   CREATE TABLE public.study_user_attributes (
    study_user_attribute_id integer NOT NULL,
    study_id integer,
    key character varying(512),
    value_json character varying(2048)
);
 )   DROP TABLE public.study_user_attributes;
       public         heap    postgres    false            �            1259    29864 1   study_user_attributes_study_user_attribute_id_seq    SEQUENCE     �   CREATE SEQUENCE public.study_user_attributes_study_user_attribute_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 H   DROP SEQUENCE public.study_user_attributes_study_user_attribute_id_seq;
       public          postgres    false    206            J           0    0 1   study_user_attributes_study_user_attribute_id_seq    SEQUENCE OWNED BY     �   ALTER SEQUENCE public.study_user_attributes_study_user_attribute_id_seq OWNED BY public.study_user_attributes.study_user_attribute_id;
          public          postgres    false    205            �            1259    29984    trial_intermediate_values    TABLE     �   CREATE TABLE public.trial_intermediate_values (
    trial_intermediate_value_id integer NOT NULL,
    trial_id integer NOT NULL,
    step integer NOT NULL,
    intermediate_value double precision NOT NULL
);
 -   DROP TABLE public.trial_intermediate_values;
       public         heap    postgres    false            �            1259    29982 9   trial_intermediate_values_trial_intermediate_value_id_seq    SEQUENCE     �   CREATE SEQUENCE public.trial_intermediate_values_trial_intermediate_value_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 P   DROP SEQUENCE public.trial_intermediate_values_trial_intermediate_value_id_seq;
       public          postgres    false    220            K           0    0 9   trial_intermediate_values_trial_intermediate_value_id_seq    SEQUENCE OWNED BY     �   ALTER SEQUENCE public.trial_intermediate_values_trial_intermediate_value_id_seq OWNED BY public.trial_intermediate_values.trial_intermediate_value_id;
          public          postgres    false    219            �            1259    29951    trial_params    TABLE     �   CREATE TABLE public.trial_params (
    param_id integer NOT NULL,
    trial_id integer,
    param_name character varying(512),
    param_value double precision,
    distribution_json character varying(2048)
);
     DROP TABLE public.trial_params;
       public         heap    postgres    false            �            1259    29949    trial_params_param_id_seq    SEQUENCE     �   CREATE SEQUENCE public.trial_params_param_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 0   DROP SEQUENCE public.trial_params_param_id_seq;
       public          postgres    false    216            L           0    0    trial_params_param_id_seq    SEQUENCE OWNED BY     W   ALTER SEQUENCE public.trial_params_param_id_seq OWNED BY public.trial_params.param_id;
          public          postgres    false    215            �            1259    29933    trial_system_attributes    TABLE     �   CREATE TABLE public.trial_system_attributes (
    trial_system_attribute_id integer NOT NULL,
    trial_id integer,
    key character varying(512),
    value_json character varying(2048)
);
 +   DROP TABLE public.trial_system_attributes;
       public         heap    postgres    false            �            1259    29931 5   trial_system_attributes_trial_system_attribute_id_seq    SEQUENCE     �   CREATE SEQUENCE public.trial_system_attributes_trial_system_attribute_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 L   DROP SEQUENCE public.trial_system_attributes_trial_system_attribute_id_seq;
       public          postgres    false    214            M           0    0 5   trial_system_attributes_trial_system_attribute_id_seq    SEQUENCE OWNED BY     �   ALTER SEQUENCE public.trial_system_attributes_trial_system_attribute_id_seq OWNED BY public.trial_system_attributes.trial_system_attribute_id;
          public          postgres    false    213            �            1259    29915    trial_user_attributes    TABLE     �   CREATE TABLE public.trial_user_attributes (
    trial_user_attribute_id integer NOT NULL,
    trial_id integer,
    key character varying(512),
    value_json character varying(2048)
);
 )   DROP TABLE public.trial_user_attributes;
       public         heap    postgres    false            �            1259    29913 1   trial_user_attributes_trial_user_attribute_id_seq    SEQUENCE     �   CREATE SEQUENCE public.trial_user_attributes_trial_user_attribute_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 H   DROP SEQUENCE public.trial_user_attributes_trial_user_attribute_id_seq;
       public          postgres    false    212            N           0    0 1   trial_user_attributes_trial_user_attribute_id_seq    SEQUENCE OWNED BY     �   ALTER SEQUENCE public.trial_user_attributes_trial_user_attribute_id_seq OWNED BY public.trial_user_attributes.trial_user_attribute_id;
          public          postgres    false    211            �            1259    29969    trial_values    TABLE     �   CREATE TABLE public.trial_values (
    trial_value_id integer NOT NULL,
    trial_id integer NOT NULL,
    objective integer NOT NULL,
    value double precision NOT NULL
);
     DROP TABLE public.trial_values;
       public         heap    postgres    false            �            1259    29967    trial_values_trial_value_id_seq    SEQUENCE     �   CREATE SEQUENCE public.trial_values_trial_value_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 6   DROP SEQUENCE public.trial_values_trial_value_id_seq;
       public          postgres    false    218            O           0    0    trial_values_trial_value_id_seq    SEQUENCE OWNED BY     c   ALTER SEQUENCE public.trial_values_trial_value_id_seq OWNED BY public.trial_values.trial_value_id;
          public          postgres    false    217            �            1259    29902    trials    TABLE     �   CREATE TABLE public.trials (
    trial_id integer NOT NULL,
    number integer,
    study_id integer,
    state public.trialstate NOT NULL,
    datetime_start timestamp without time zone,
    datetime_complete timestamp without time zone
);
    DROP TABLE public.trials;
       public         heap    postgres    false    647            �            1259    29900    trials_trial_id_seq    SEQUENCE     �   CREATE SEQUENCE public.trials_trial_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 *   DROP SEQUENCE public.trials_trial_id_seq;
       public          postgres    false    210            P           0    0    trials_trial_id_seq    SEQUENCE OWNED BY     K   ALTER SEQUENCE public.trials_trial_id_seq OWNED BY public.trials.trial_id;
          public          postgres    false    209            �            1259    29843    version_info    TABLE     �   CREATE TABLE public.version_info (
    version_info_id integer NOT NULL,
    schema_version integer,
    library_version character varying(256),
    CONSTRAINT version_info_version_info_id_check CHECK ((version_info_id = 1))
);
     DROP TABLE public.version_info;
       public         heap    postgres    false            l           2604    29836    studies study_id    DEFAULT     t   ALTER TABLE ONLY public.studies ALTER COLUMN study_id SET DEFAULT nextval('public.studies_study_id_seq'::regclass);
 ?   ALTER TABLE public.studies ALTER COLUMN study_id DROP DEFAULT;
       public          postgres    false    201    200    201            n           2604    29854 #   study_directions study_direction_id    DEFAULT     �   ALTER TABLE ONLY public.study_directions ALTER COLUMN study_direction_id SET DEFAULT nextval('public.study_directions_study_direction_id_seq'::regclass);
 R   ALTER TABLE public.study_directions ALTER COLUMN study_direction_id DROP DEFAULT;
       public          postgres    false    204    203    204            p           2604    29887 1   study_system_attributes study_system_attribute_id    DEFAULT     �   ALTER TABLE ONLY public.study_system_attributes ALTER COLUMN study_system_attribute_id SET DEFAULT nextval('public.study_system_attributes_study_system_attribute_id_seq'::regclass);
 `   ALTER TABLE public.study_system_attributes ALTER COLUMN study_system_attribute_id DROP DEFAULT;
       public          postgres    false    208    207    208            o           2604    29869 -   study_user_attributes study_user_attribute_id    DEFAULT     �   ALTER TABLE ONLY public.study_user_attributes ALTER COLUMN study_user_attribute_id SET DEFAULT nextval('public.study_user_attributes_study_user_attribute_id_seq'::regclass);
 \   ALTER TABLE public.study_user_attributes ALTER COLUMN study_user_attribute_id DROP DEFAULT;
       public          postgres    false    205    206    206            v           2604    29987 5   trial_intermediate_values trial_intermediate_value_id    DEFAULT     �   ALTER TABLE ONLY public.trial_intermediate_values ALTER COLUMN trial_intermediate_value_id SET DEFAULT nextval('public.trial_intermediate_values_trial_intermediate_value_id_seq'::regclass);
 d   ALTER TABLE public.trial_intermediate_values ALTER COLUMN trial_intermediate_value_id DROP DEFAULT;
       public          postgres    false    219    220    220            t           2604    29954    trial_params param_id    DEFAULT     ~   ALTER TABLE ONLY public.trial_params ALTER COLUMN param_id SET DEFAULT nextval('public.trial_params_param_id_seq'::regclass);
 D   ALTER TABLE public.trial_params ALTER COLUMN param_id DROP DEFAULT;
       public          postgres    false    216    215    216            s           2604    29936 1   trial_system_attributes trial_system_attribute_id    DEFAULT     �   ALTER TABLE ONLY public.trial_system_attributes ALTER COLUMN trial_system_attribute_id SET DEFAULT nextval('public.trial_system_attributes_trial_system_attribute_id_seq'::regclass);
 `   ALTER TABLE public.trial_system_attributes ALTER COLUMN trial_system_attribute_id DROP DEFAULT;
       public          postgres    false    213    214    214            r           2604    29918 -   trial_user_attributes trial_user_attribute_id    DEFAULT     �   ALTER TABLE ONLY public.trial_user_attributes ALTER COLUMN trial_user_attribute_id SET DEFAULT nextval('public.trial_user_attributes_trial_user_attribute_id_seq'::regclass);
 \   ALTER TABLE public.trial_user_attributes ALTER COLUMN trial_user_attribute_id DROP DEFAULT;
       public          postgres    false    212    211    212            u           2604    29972    trial_values trial_value_id    DEFAULT     �   ALTER TABLE ONLY public.trial_values ALTER COLUMN trial_value_id SET DEFAULT nextval('public.trial_values_trial_value_id_seq'::regclass);
 J   ALTER TABLE public.trial_values ALTER COLUMN trial_value_id DROP DEFAULT;
       public          postgres    false    217    218    218            q           2604    29905    trials trial_id    DEFAULT     r   ALTER TABLE ONLY public.trials ALTER COLUMN trial_id SET DEFAULT nextval('public.trials_trial_id_seq'::regclass);
 >   ALTER TABLE public.trials ALTER COLUMN trial_id DROP DEFAULT;
       public          postgres    false    209    210    210            @          0    29997    alembic_version 
   TABLE DATA           6   COPY public.alembic_version (version_num) FROM stdin;
    public          postgres    false    221   g�       ,          0    29833    studies 
   TABLE DATA           7   COPY public.studies (study_id, study_name) FROM stdin;
    public          postgres    false    201   ��       /          0    29851    study_directions 
   TABLE DATA           ^   COPY public.study_directions (study_direction_id, direction, study_id, objective) FROM stdin;
    public          postgres    false    204   �       3          0    29884    study_system_attributes 
   TABLE DATA           g   COPY public.study_system_attributes (study_system_attribute_id, study_id, key, value_json) FROM stdin;
    public          postgres    false    208   j�       1          0    29866    study_user_attributes 
   TABLE DATA           c   COPY public.study_user_attributes (study_user_attribute_id, study_id, key, value_json) FROM stdin;
    public          postgres    false    206   ��       ?          0    29984    trial_intermediate_values 
   TABLE DATA           t   COPY public.trial_intermediate_values (trial_intermediate_value_id, trial_id, step, intermediate_value) FROM stdin;
    public          postgres    false    220   ��       ;          0    29951    trial_params 
   TABLE DATA           f   COPY public.trial_params (param_id, trial_id, param_name, param_value, distribution_json) FROM stdin;
    public          postgres    false    216   ��       9          0    29933    trial_system_attributes 
   TABLE DATA           g   COPY public.trial_system_attributes (trial_system_attribute_id, trial_id, key, value_json) FROM stdin;
    public          postgres    false    214   g!      7          0    29915    trial_user_attributes 
   TABLE DATA           c   COPY public.trial_user_attributes (trial_user_attribute_id, trial_id, key, value_json) FROM stdin;
    public          postgres    false    212   �!      =          0    29969    trial_values 
   TABLE DATA           R   COPY public.trial_values (trial_value_id, trial_id, objective, value) FROM stdin;
    public          postgres    false    218   �!      5          0    29902    trials 
   TABLE DATA           f   COPY public.trials (trial_id, number, study_id, state, datetime_start, datetime_complete) FROM stdin;
    public          postgres    false    210   4      -          0    29843    version_info 
   TABLE DATA           X   COPY public.version_info (version_info_id, schema_version, library_version) FROM stdin;
    public          postgres    false    202   �O      Q           0    0    studies_study_id_seq    SEQUENCE SET     C   SELECT pg_catalog.setval('public.studies_study_id_seq', 31, true);
          public          postgres    false    200            R           0    0 '   study_directions_study_direction_id_seq    SEQUENCE SET     V   SELECT pg_catalog.setval('public.study_directions_study_direction_id_seq', 12, true);
          public          postgres    false    203            S           0    0 5   study_system_attributes_study_system_attribute_id_seq    SEQUENCE SET     d   SELECT pg_catalog.setval('public.study_system_attributes_study_system_attribute_id_seq', 1, false);
          public          postgres    false    207            T           0    0 1   study_user_attributes_study_user_attribute_id_seq    SEQUENCE SET     `   SELECT pg_catalog.setval('public.study_user_attributes_study_user_attribute_id_seq', 1, false);
          public          postgres    false    205            U           0    0 9   trial_intermediate_values_trial_intermediate_value_id_seq    SEQUENCE SET     i   SELECT pg_catalog.setval('public.trial_intermediate_values_trial_intermediate_value_id_seq', 327, true);
          public          postgres    false    219            V           0    0    trial_params_param_id_seq    SEQUENCE SET     J   SELECT pg_catalog.setval('public.trial_params_param_id_seq', 4746, true);
          public          postgres    false    215            W           0    0 5   trial_system_attributes_trial_system_attribute_id_seq    SEQUENCE SET     d   SELECT pg_catalog.setval('public.trial_system_attributes_trial_system_attribute_id_seq', 1, false);
          public          postgres    false    213            X           0    0 1   trial_user_attributes_trial_user_attribute_id_seq    SEQUENCE SET     `   SELECT pg_catalog.setval('public.trial_user_attributes_trial_user_attribute_id_seq', 1, false);
          public          postgres    false    211            Y           0    0    trial_values_trial_value_id_seq    SEQUENCE SET     O   SELECT pg_catalog.setval('public.trial_values_trial_value_id_seq', 327, true);
          public          postgres    false    217            Z           0    0    trials_trial_id_seq    SEQUENCE SET     C   SELECT pg_catalog.setval('public.trials_trial_id_seq', 358, true);
          public          postgres    false    209            �           2606    30001 #   alembic_version alembic_version_pkc 
   CONSTRAINT     j   ALTER TABLE ONLY public.alembic_version
    ADD CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num);
 M   ALTER TABLE ONLY public.alembic_version DROP CONSTRAINT alembic_version_pkc;
       public            postgres    false    221            y           2606    29841    studies studies_pkey 
   CONSTRAINT     X   ALTER TABLE ONLY public.studies
    ADD CONSTRAINT studies_pkey PRIMARY KEY (study_id);
 >   ALTER TABLE ONLY public.studies DROP CONSTRAINT studies_pkey;
       public            postgres    false    201            }           2606    29856 &   study_directions study_directions_pkey 
   CONSTRAINT     t   ALTER TABLE ONLY public.study_directions
    ADD CONSTRAINT study_directions_pkey PRIMARY KEY (study_direction_id);
 P   ALTER TABLE ONLY public.study_directions DROP CONSTRAINT study_directions_pkey;
       public            postgres    false    204                       2606    29858 8   study_directions study_directions_study_id_objective_key 
   CONSTRAINT     �   ALTER TABLE ONLY public.study_directions
    ADD CONSTRAINT study_directions_study_id_objective_key UNIQUE (study_id, objective);
 b   ALTER TABLE ONLY public.study_directions DROP CONSTRAINT study_directions_study_id_objective_key;
       public            postgres    false    204    204            �           2606    29892 4   study_system_attributes study_system_attributes_pkey 
   CONSTRAINT     �   ALTER TABLE ONLY public.study_system_attributes
    ADD CONSTRAINT study_system_attributes_pkey PRIMARY KEY (study_system_attribute_id);
 ^   ALTER TABLE ONLY public.study_system_attributes DROP CONSTRAINT study_system_attributes_pkey;
       public            postgres    false    208            �           2606    29894 @   study_system_attributes study_system_attributes_study_id_key_key 
   CONSTRAINT     �   ALTER TABLE ONLY public.study_system_attributes
    ADD CONSTRAINT study_system_attributes_study_id_key_key UNIQUE (study_id, key);
 j   ALTER TABLE ONLY public.study_system_attributes DROP CONSTRAINT study_system_attributes_study_id_key_key;
       public            postgres    false    208    208            �           2606    29874 0   study_user_attributes study_user_attributes_pkey 
   CONSTRAINT     �   ALTER TABLE ONLY public.study_user_attributes
    ADD CONSTRAINT study_user_attributes_pkey PRIMARY KEY (study_user_attribute_id);
 Z   ALTER TABLE ONLY public.study_user_attributes DROP CONSTRAINT study_user_attributes_pkey;
       public            postgres    false    206            �           2606    29876 <   study_user_attributes study_user_attributes_study_id_key_key 
   CONSTRAINT     �   ALTER TABLE ONLY public.study_user_attributes
    ADD CONSTRAINT study_user_attributes_study_id_key_key UNIQUE (study_id, key);
 f   ALTER TABLE ONLY public.study_user_attributes DROP CONSTRAINT study_user_attributes_study_id_key_key;
       public            postgres    false    206    206            �           2606    29989 8   trial_intermediate_values trial_intermediate_values_pkey 
   CONSTRAINT     �   ALTER TABLE ONLY public.trial_intermediate_values
    ADD CONSTRAINT trial_intermediate_values_pkey PRIMARY KEY (trial_intermediate_value_id);
 b   ALTER TABLE ONLY public.trial_intermediate_values DROP CONSTRAINT trial_intermediate_values_pkey;
       public            postgres    false    220            �           2606    29991 E   trial_intermediate_values trial_intermediate_values_trial_id_step_key 
   CONSTRAINT     �   ALTER TABLE ONLY public.trial_intermediate_values
    ADD CONSTRAINT trial_intermediate_values_trial_id_step_key UNIQUE (trial_id, step);
 o   ALTER TABLE ONLY public.trial_intermediate_values DROP CONSTRAINT trial_intermediate_values_trial_id_step_key;
       public            postgres    false    220    220            �           2606    29959    trial_params trial_params_pkey 
   CONSTRAINT     b   ALTER TABLE ONLY public.trial_params
    ADD CONSTRAINT trial_params_pkey PRIMARY KEY (param_id);
 H   ALTER TABLE ONLY public.trial_params DROP CONSTRAINT trial_params_pkey;
       public            postgres    false    216            �           2606    29961 1   trial_params trial_params_trial_id_param_name_key 
   CONSTRAINT     |   ALTER TABLE ONLY public.trial_params
    ADD CONSTRAINT trial_params_trial_id_param_name_key UNIQUE (trial_id, param_name);
 [   ALTER TABLE ONLY public.trial_params DROP CONSTRAINT trial_params_trial_id_param_name_key;
       public            postgres    false    216    216            �           2606    29941 4   trial_system_attributes trial_system_attributes_pkey 
   CONSTRAINT     �   ALTER TABLE ONLY public.trial_system_attributes
    ADD CONSTRAINT trial_system_attributes_pkey PRIMARY KEY (trial_system_attribute_id);
 ^   ALTER TABLE ONLY public.trial_system_attributes DROP CONSTRAINT trial_system_attributes_pkey;
       public            postgres    false    214            �           2606    29943 @   trial_system_attributes trial_system_attributes_trial_id_key_key 
   CONSTRAINT     �   ALTER TABLE ONLY public.trial_system_attributes
    ADD CONSTRAINT trial_system_attributes_trial_id_key_key UNIQUE (trial_id, key);
 j   ALTER TABLE ONLY public.trial_system_attributes DROP CONSTRAINT trial_system_attributes_trial_id_key_key;
       public            postgres    false    214    214            �           2606    29923 0   trial_user_attributes trial_user_attributes_pkey 
   CONSTRAINT     �   ALTER TABLE ONLY public.trial_user_attributes
    ADD CONSTRAINT trial_user_attributes_pkey PRIMARY KEY (trial_user_attribute_id);
 Z   ALTER TABLE ONLY public.trial_user_attributes DROP CONSTRAINT trial_user_attributes_pkey;
       public            postgres    false    212            �           2606    29925 <   trial_user_attributes trial_user_attributes_trial_id_key_key 
   CONSTRAINT     �   ALTER TABLE ONLY public.trial_user_attributes
    ADD CONSTRAINT trial_user_attributes_trial_id_key_key UNIQUE (trial_id, key);
 f   ALTER TABLE ONLY public.trial_user_attributes DROP CONSTRAINT trial_user_attributes_trial_id_key_key;
       public            postgres    false    212    212            �           2606    29974    trial_values trial_values_pkey 
   CONSTRAINT     h   ALTER TABLE ONLY public.trial_values
    ADD CONSTRAINT trial_values_pkey PRIMARY KEY (trial_value_id);
 H   ALTER TABLE ONLY public.trial_values DROP CONSTRAINT trial_values_pkey;
       public            postgres    false    218            �           2606    29976 0   trial_values trial_values_trial_id_objective_key 
   CONSTRAINT     z   ALTER TABLE ONLY public.trial_values
    ADD CONSTRAINT trial_values_trial_id_objective_key UNIQUE (trial_id, objective);
 Z   ALTER TABLE ONLY public.trial_values DROP CONSTRAINT trial_values_trial_id_objective_key;
       public            postgres    false    218    218            �           2606    29907    trials trials_pkey 
   CONSTRAINT     V   ALTER TABLE ONLY public.trials
    ADD CONSTRAINT trials_pkey PRIMARY KEY (trial_id);
 <   ALTER TABLE ONLY public.trials DROP CONSTRAINT trials_pkey;
       public            postgres    false    210            {           2606    29848    version_info version_info_pkey 
   CONSTRAINT     i   ALTER TABLE ONLY public.version_info
    ADD CONSTRAINT version_info_pkey PRIMARY KEY (version_info_id);
 H   ALTER TABLE ONLY public.version_info DROP CONSTRAINT version_info_pkey;
       public            postgres    false    202            w           1259    29842    ix_studies_study_name    INDEX     V   CREATE UNIQUE INDEX ix_studies_study_name ON public.studies USING btree (study_name);
 )   DROP INDEX public.ix_studies_study_name;
       public            postgres    false    201            �           2606    29859 /   study_directions study_directions_study_id_fkey    FK CONSTRAINT     �   ALTER TABLE ONLY public.study_directions
    ADD CONSTRAINT study_directions_study_id_fkey FOREIGN KEY (study_id) REFERENCES public.studies(study_id);
 Y   ALTER TABLE ONLY public.study_directions DROP CONSTRAINT study_directions_study_id_fkey;
       public          postgres    false    201    2937    204            �           2606    29895 =   study_system_attributes study_system_attributes_study_id_fkey    FK CONSTRAINT     �   ALTER TABLE ONLY public.study_system_attributes
    ADD CONSTRAINT study_system_attributes_study_id_fkey FOREIGN KEY (study_id) REFERENCES public.studies(study_id);
 g   ALTER TABLE ONLY public.study_system_attributes DROP CONSTRAINT study_system_attributes_study_id_fkey;
       public          postgres    false    208    201    2937            �           2606    29877 9   study_user_attributes study_user_attributes_study_id_fkey    FK CONSTRAINT     �   ALTER TABLE ONLY public.study_user_attributes
    ADD CONSTRAINT study_user_attributes_study_id_fkey FOREIGN KEY (study_id) REFERENCES public.studies(study_id);
 c   ALTER TABLE ONLY public.study_user_attributes DROP CONSTRAINT study_user_attributes_study_id_fkey;
       public          postgres    false    206    2937    201            �           2606    29992 A   trial_intermediate_values trial_intermediate_values_trial_id_fkey    FK CONSTRAINT     �   ALTER TABLE ONLY public.trial_intermediate_values
    ADD CONSTRAINT trial_intermediate_values_trial_id_fkey FOREIGN KEY (trial_id) REFERENCES public.trials(trial_id);
 k   ALTER TABLE ONLY public.trial_intermediate_values DROP CONSTRAINT trial_intermediate_values_trial_id_fkey;
       public          postgres    false    2953    210    220            �           2606    29962 '   trial_params trial_params_trial_id_fkey    FK CONSTRAINT     �   ALTER TABLE ONLY public.trial_params
    ADD CONSTRAINT trial_params_trial_id_fkey FOREIGN KEY (trial_id) REFERENCES public.trials(trial_id);
 Q   ALTER TABLE ONLY public.trial_params DROP CONSTRAINT trial_params_trial_id_fkey;
       public          postgres    false    216    2953    210            �           2606    29944 =   trial_system_attributes trial_system_attributes_trial_id_fkey    FK CONSTRAINT     �   ALTER TABLE ONLY public.trial_system_attributes
    ADD CONSTRAINT trial_system_attributes_trial_id_fkey FOREIGN KEY (trial_id) REFERENCES public.trials(trial_id);
 g   ALTER TABLE ONLY public.trial_system_attributes DROP CONSTRAINT trial_system_attributes_trial_id_fkey;
       public          postgres    false    214    2953    210            �           2606    29926 9   trial_user_attributes trial_user_attributes_trial_id_fkey    FK CONSTRAINT     �   ALTER TABLE ONLY public.trial_user_attributes
    ADD CONSTRAINT trial_user_attributes_trial_id_fkey FOREIGN KEY (trial_id) REFERENCES public.trials(trial_id);
 c   ALTER TABLE ONLY public.trial_user_attributes DROP CONSTRAINT trial_user_attributes_trial_id_fkey;
       public          postgres    false    2953    212    210            �           2606    29977 '   trial_values trial_values_trial_id_fkey    FK CONSTRAINT     �   ALTER TABLE ONLY public.trial_values
    ADD CONSTRAINT trial_values_trial_id_fkey FOREIGN KEY (trial_id) REFERENCES public.trials(trial_id);
 Q   ALTER TABLE ONLY public.trial_values DROP CONSTRAINT trial_values_trial_id_fkey;
       public          postgres    false    210    218    2953            �           2606    29908    trials trials_study_id_fkey    FK CONSTRAINT     �   ALTER TABLE ONLY public.trials
    ADD CONSTRAINT trials_study_id_fkey FOREIGN KEY (study_id) REFERENCES public.studies(study_id);
 E   ALTER TABLE ONLY public.trials DROP CONSTRAINT trials_study_id_fkey;
       public          postgres    false    201    210    2937            @      x�+3�3�3�K����� ��      ,   g   x�e�K
� ��],$
�0�E�i�~�޾A%�ü�S*ƛ���|D6�_�S	�0��-?'8e
Dq�T݀[�O��Eۂ	���x�:me�Z��\�� �hQ�      /   V   x�M�1� D�:9��
�����a�b˗L�6�g���K�s�,�`]��]n�>��7qwq�cI�2`����?м(N      3      x������ � �      1      x������ � �      ?      x�E�[�l)���T��^f��hw �uGH�PHT�ħ|�ʷ��z��ڌ�wO��|b|����,��n�~�ݒs��k��GYy���n��Zrm�>�hk��)���5FkcVN�='e�o^�v~^b��r$�:�����QV�Z{Z����}��v���-�Oi϶6Y�=g�UV�3���}��]�K�<R��ywqx-m�ZF���	��z��Ĉ9
K�h�t~�a��W�F�{��U�h�}��	ˣܵh}j����,�xw�Z]�`�;j�	���U�dDr���"�Z����,���K�9�G�1�r�Ğ��Q�Įo�ze$��\�gu�51v&��3?p�.�FpI�*pa�bbn�9�=�>�G���/GLa�?5?��rх=;k�S�5���W��ξ�W}U˷,n\GnKv�Omwۘ��1�80MXQ�]�X����,ؓ�����VZ���X�l#aF}.����2b;�=fT�V�w {>���8;qZwW)�q̖��',�󮭽Ü%�f����a���{�
Z�6�|�5Ӆ��n�J�k�$& �p���z��iz��o���sܼk���q"C:�����}'>�n�������E¸v�O�νbt":�C��V �W��JO�t�\��GU�%���x��d�kJm�&�t�xw����(b	��?��N� #�D�*�����q=�ً{`��t}���x�O�R۟�υ�����%���[�c~@ au_3�2F��h�'l��};|�K� �E²��*y^r�P�snH�Ƌ.�Y���Z)���(��|�D�q�	��K��HU<_4o~�e��,m�g}��D�B����R֨|h�8~���7b��a��q�h3$Z'�0Kg¶�^���Qsp��%l��;�|��w�Z=a��G_R1��U�6	�w�<T󠷒�`�Kx��B$-���̗�� ��L�
 ?&�y�F�X�h��Xb��LM/�@]My/r���Y' ��'9@���[��35�wX���燶�Y>�8U�Y�αJ����\j�%�ؗ�U?� v}؟�I�I�r�կ�pA��a��.����Nܖ���Q(q��.�I�-�|z�)9?;�ڤh��(O��	?��h	6�ȁ	@��	ֹ��X��PM�K{ИQX� V�-��
,��`w�vIo�[��	�	�a��Ֆ�G.7!ɩ��+u)�� @�w kCz���G}&����u����[j��P�0 �.5ת�Y���ݘ�7d�D
��r�<.0�jG3��&���=�Z¯A�� �'�Q�d�� ����X( ���������
憛��׾��=��d(h���ɠJV���U)�6{�j��z�'{����Uv��`�Q��Ó�+��p�"A
�ǒ�x�	m��0��� KB∓���^RT�is�t�PG� iD,NM@��# �0��M����N௮i_1���$��҉_<�"���0���ȗ��P1n"b5�_�5�O���:�1�w��er�`���n���<�[)��R��c,�9rjR��]�)+%�'Z�F!������N�9C�&I擢�ȟ8��bJ55�p�^;:�Of��x�j'��2P��'��\���9������U��j�������t�s��ߒ�;UN� 3��J��/[���♫��M.�".@�DܑV�d��lH&��J�F�T7���Mau�X�����5��(�W?(�D���Aa�[��WH-
�9�`���A]B�T:��u���F�HJ0��K!>�l�����|����E� .HKҖ�`L]/zھp�P?؅X������*T7���k|Dy�2��?ݸ@rkd��^_�t�{��),_G)����/�]��'���gq�E��N���AXu8���~,��b}>ӽ {N�����#��u\�ԇR�����:l�p"�LHG>4N�S�4K�O�h��Qu�J�5\%|�]G,ܯ��z G:0;Z���{�(�=D���j"|�5�v�6q��Q_F�5(�(q�U��W��!j�*�NP�j�SF�=��G���-g��rBw��U��r�C���; <�����Klɭ,`��Q]�8N����ٹr�,�(tNM�$$Q6o���ec�����J?�MFZ�9�pC��o�G i���
����$Z�,
%�q�+n�X��Gɓ5ؕ8��J=d���
�t�wuJV�����z�w��:uI����1��:sIC�g��@W�vY�w�$�������V���Rhw�BC���³�J ������]�tT���肄vh�9�2�fRxpC7O��b\
��$ő:�R�����ׁ�(�P�N�u��c�m�ñC�G"�iO�}��ۀ?! �b�O)�f�L�j&��}�6�3v�L�`��%ѕ+��T~v�@s�6j,*�|UG:�ۯ�D�V$+	������.Yf�����Xv�$@;����!��A�կ%Jb�8�2�G�B��74�@ۙ�8�y^Xh;@eX`��ޠ~�G�mv`�Dd^1M?��>?��r"J"?Y�WY�`Џ�F`3���F����07Cq�V7�	�� ����LGEKo$�V_!F Xy�tOd�]Ԋc�`�Gʕ�`N��������p5��A_ql�6�7Ma�����F]0a]�,UD�}q�1[�]m_�1��z���i����=R 苫�Ib˷G�&��W��A'E���V���.�����.�����ޯ�o���o�|q$��6$褝�IxE��:�:��P�dk�lF�1$�������h^��U��%�pbC��8�B ����?�{����6&up#Љ���5tP:��O���$��#Ѵ4�I��˴��
e
�tGs`3�p���J�Q�6?� �֦��aL��2�'��z�EY�q����U�t���i�X��������ގF#�>�*f����} L�%��ɠ�$N��0t�U�ՁC�u�j#:�)���ժ�%�I����Hx(L�o���⥫��(�\*8]������7l5�rR\�I�:�~AN˃N?��;P��e`:r|�Uь,��F��DH���t�hۄ���N�GN��#m��J�vH;�ThF{�4a֘��p��W��.�G��]p:1n��S�e�
�<+	͸r�,�HB�a!ד،����q{|Χ�'�G�ӏ-(�&.N�K3�'3�o�A)Λ�e�q#6&��c�Ua���1ۘ%�;�H�2�����S8 �&������~E0Gc��+�~���Ne������r�#f���9x���7r���` �śY�D9A�T�  �x�#��%�'| &c���3lYV��L�2��ht^C9Ad;�;9L�?%\��H�Vvч��P!�9r�$R^|� v1^�!��Y-�>��I�\6��In:�J9���mD�$c�NK,��������dj@��݋���hv�7��6��G���N��)�(�\܂ʘ?��D��|��g�o2��FY������h<Տ�Gi��z0Ќ�j੽91�W�����cd�}��qUv9c�L�|ѭ�i���"�3E�Q� �1߸FE�O=d���@g�'�Pb벼�).�u��ڬ>D���&C�!+?P�}�����C	S�F��{�x�GْҮ: ��M�3��_���0�����3��ǖ�/�B�LX�:|�������0�B;9��_�ޏZ�)��sqӑ�ÑT�x���}!��د7R����7[=']I�q8S���nO�3��X,V�|�l�<���wXI�:͚*Lʢ�g��pk�שѩ���~ @���̊3Nsf�ϸ�7EM��N��S�ǔX�MA�<ԇ�;�@&����U���� ��3���nu�5�� |��й�y\,�Ӕ�0�x�� �zŰ5I��}&JB�^e** C�d�
I+4k��nu씝���Zp�~��o'��5<���Awd�AXl��(_��@ܠY�9�{L&�Z�L�r7�uS��U���z>��y2z�'A�^��R�c���z� >  *�3��(�ƑkA�>VW����Z�y9�_��(+���o���~��Q(r�ͳ��ʿ�s��2-���5�����tN.�+������	E�E*�����
h�>Z	��JPV��n�V�$�^���.�D�BTPl�����=S5�A	�령�#��3 ����&�\��o���h)��.��"��/�3~��Ҍ{�{�T��l�
%�\�{u"�>")�՝L�Z��t���d"�k���$����X�?Wp"�qé��
����=�})�����C���~�{G�����x0>�b�|���3�`���泖S�,���P�y��sO�9є�j��L���H��LM���yNKfB���謎J�����Xm�%�1j#��Á�)VSJ Gn�ܠi,.j��Sэ>�u�@�	V��Ca��ךk��	B�� 򃭻u��������A�>�PV̮�_�Y�{�m�?vq�Wėb���S��IfB}��\Ɋ�3�Q��_��v}�+��v֚�_]��e|Φ��$P
���o���0$8ُd��7��wY	3
\U\��O��gYsw^��M)��i��      ;      x�̽_��ɑ����勪�S��7�~1`�Oz2b��⒫��-軻"k������og�ٕ����w΍�efd��2������������������w�ӿ����ǿ���~��������/?���۟~�����?��Ͽ����w��t��������ݟ��_��������������?�����������������B�����~˯���$����������1�������?��?��������e|��|�����?�ӯ������?�_�����O����d��������������������˟�_]������?}�E>�������f���z���O~����w@^�p~���?������7���6�`%ˌ��W����/��[>4���.u�������?����������7��s�n��;8�4��t������� �g����|��P�����O����_?��J����)q�ؿ�O?~��?��?���~�������%s?=�o�G7i�{�	8�����1��oA~����]����~�ݏ��o�3�������M��w�������?��|�~���#��+�n�<|���9��_�����7G�������ß��������#��O��?���?���;�O����}�,�/��������������3�S�ϵ���9J��)������g�_�����ⷱ?|@��ޘDb�0^ʃ��ߞ�in���������~���3��@���,T ���h��
�i��u �F���A>��N�rI(����h����(�׼��R� L�N��/`5+ >��7:l��9�2`SxA���;* ^ �)ؾ~��
���:�AP�o���k
 `�p�K)��[.)��.�j�?"��� �����@l��w:@�����BhŰEk�U� �+�����S  �Q �_?Ο��o��@i}� �����(Ȣ��`�Pp�Ku��rI��+gmH}C�2 �Q �R� kl�:Xs��f ��A�M�x��T���x#w(� �(�үga���J>��k
�@M5�k�(�W������b�
`�!~�jF���TL6B���_�N�>��c�k���b�ls`��`�	���y>6�W��?ӯI��-�ڨ� Y<H����'%�A.�$���Z��_D`���PJU�u2��_�����A�m�-	&'v��nPL`w��g&��Hˁ }%��R��D�FSAI 5rI/5���K����m����_AJa�	�Z�� `Z���§��(
sH~P�D�M��c�����fcw�s���氇���.|�C���sT0G����_���d�"��L�x)m�3��q>Z*6MXc�|o`NL�:�Zg�H����`N���D!�Sl�h�Y3>�3���:|�m��:�H�&u���PV��+���&���&%�4u�&%��%��0aR��-u�MFն���ȿ�C��tR �Vx �Px���7x�m���������Uލ]�
&��K[��+u�4ֿ��O�Rx�uٌ�@�ߠ��Uu�d����$��RI¹���H��R�B )_<L�<� �PDp��$+?�y�755!�p��h�`�&��7l�K��yÔxx���P����]��&���&}�VcI�n�m$+� ť�.�T�Q�`j�A���K��5�z��f2�Kj�h3a�	��Ң�jj,-~x��)K ��XZI ]값I��a%��Ei��[������@����&`j����`�K��I~PO�+��c��A��x0xK�`+��:KfO��x0�W��i�ۯʃ%�tyRI�<x�ku�<	�����1�]K�V�w�n M�Z� 4xG��uCXB\
B�w�w������b�1]�������Dh��&�v�xoo�T��;Fo���J��u�z�W�!���CBפ��(O�#����{O�*Tۚ�(56
�Aפ�x67%��X�6'	�ѬP� FOJ��֥��G��	U���16�!��A6�a�*��)�5� ;t�':y��x����Z�ܾ?v��ɸ ���c��Ğc��Ċ����?���[ˏn�KP�׸�&���6�Vz��H�QC�A�-�Ol>�_/��>Z�FHW�-�I�40�?^K#���aqҫ[nX����	K�^���G/T¶�_��k��.M�9���b�e�cM���!�=��؇��J������bQ��~��4�@��k4�I���G��KM�i&�����dT_�.d�g���X����mǸT;� �}�8���Ȗ��k�?a2�t]��d����h���@��ȉ�I�ԁ�?^�v J����)2�g.�c"��v )!�
�!
 �I�"�A=�3�
݅�Oc�E�O`��SB�t�`����t����8	�i��U���"C/2��!,h�Y�H��Q�(�x&���QEEX���F9a'k<+����!���W	K[c6����5բt&����JE����ͥ�TڢG��KY���ss��A�f�v��MƋ��"U��/�˿���\�,ҵ��\�@գ��i	T�^�@3�j_,�\j�[0�2]*#7����ILFR��w���#[��2��,�f����Ƴ�G[�z��^���uZrH�m�T�T)�Ny����]��X�b�.,��X���W��,X��4�f���ײh�{��Fs��&<���2�yU��l}�H�`���H���X��}(M�E��`�c��,]qH�+�x�a�*��)�UX�8tK9��roM���ۼ���`e��a[fq��/��N��%Q�-�[Rq��-K���QI��C�W҇]�#E�j�9��'���W�Y����PoY����qa�CU%!f7�bq���͸j=�|�9.�!w���xΈ��Q�=Gx=rm��-�9!�TB�甊��}&z˶�U13�J��X��)t5sF4.�4���!��7u��.UsF?��Z#(��Aה#/5j�r�QƆF"���F"����F"���k�D�C�6'�·;�z��AQ^e�&�ȫ��'y��}'�F:�cs'��J�t�8�RuaRy-��qH�t0�Qƪ��.�X�W�k(�:Д��2J#M'c݊W`����XSS3^�Cb�"�F�DoQ����K6e\��MK��q�k4��2RiRXa���,����;���m��Fac] ����6�ok����@\��n<�d�iSF��h ����m��C_�%����J�� ՙ֏�Bj;�B�:�B��2Z!����Bx!U�1#����b����
�Ԫ�`��M u���ߦF�A=�rqr��s+�~��"y0x�'q���\��o��T+�;m��}�,m���TR$��R=̷蚗	�-�j�@�c%1΢�+�XT9n�Ċ�+�W~P4�c���<<��(��b�E=��0���9+�S�s�3*ꝱbZEҖI%E�I �U=D�-y��FթØ]��7�1����aL��2uS)�K�%�����a�;�u+���˿�:`~Eֲ���Y�:����Z�[�!��J>=X�5u��Ҵ��#��Fx$��g�G�͵�)�^G��a�{���)2^G��a�[
���)��ʆ��2?��)���5�r8�~
U��I�{���m����-��"�upo8��*mΛy��y3���p���!�k�����Z~��{:�ZCi������6l0?`]pQO�\0Q4D�D�-?Ue��_��-�Xpѵ\ڹ����o��n����
�@�W�].�q���Bz������%��)�S��pL�E.6|$|���B���]�5�mi��p27K�:���v�֡�*�X�T^L:�!�kҁ�^�s���_��*��\m�^X��R[��R�[�aر�g[�|S#�[B��H8+�d�6lV𣭻2Z��
�&[��!�Ͷn�S�.��8�R��2��Դ�;    �b�Fǎ��c7k��h�Q�;���zq�܊h���;zS_���Q
�"�Fێz١Gm�������
;
�ޚ�$c�����"6������7}��)	��{=�H�� ��F��Q�?�ߨr��n��܊�CL���c��E$x/��g�hҖ�"S�f?�����u�&m]ubNiz�gN&)RN�x-u��[��M�=�5�U>,`�e7|a���FL�U< ?�ճ��p"a[�"�6�њ�G����)-3�3�7y*�ə���KH�YQ�I��A�?^K4��: r��6#C3�c��G���G�Yq�ʎ�g���-թc�U/�1�!��Ɏ0�G�Q�����:b�?Dk��W�G���-��F��>^jd�+��8,���)	�@�K�e�����(� ���6�z������E�aS�	�q0J������VV&��\�eK��:XI�8X��kU�tsKlsQ�y��b'�J�68�P�p�pp���i&-.!��iV��+�[fҾ��tp{x��K��j]��$�.u�d�"u�$��R�H¹��و��9���׏��8�K��c�����1�}�����S�����I����MU�1�@"�<�<.�d�d=]<.�I˘_��#����qycIFH�:�H&�Q�/��&#��:��d$�+l2 ��j20�� s �NdN �q,ɤ��>�W[d+��Hxj�<�d���Ô��)d��S-�;���b镇��%3��Hf�KBI8�t�\�ʞ�s�y�јn=tF���{&����E����jPqlK�T��&��؄}1?��Y}���.H���/���7�cL���"��g[[$#�zϾ�a���"9�R%rH�ŤC	][,�A���4E�7�%P��7^?J�5"�VX�5>�l�B�k�UY!x�x���L�}�of�������7�$X��f��튙��.UY!|8奌_�wu8��/<��݆�W�	T�6�R9l&�-�0:سr���a6ȨhԿ�9�{���uI���-�[	�+vX�M:��K�t�!��?$tK:,ի�[*d}b�������Z=#H�b�?/F�H� w�t�j~^���}N]���ǅM��%~�56k�Dl����_9�57�_�/1L����5�E8	�k"��$E1N�x��'�\��~5���������)!�@���D��H$ĥr����=�o�X�d���_-�NJ�L��j[�	�T�a����VVc�+X�z�Įq�6UcX/^�lF���nLl�C���L���z`�C�S�b�@sd�#B+d�cby,?���7��4G��5%����{����Ǵ��C#X'{x��ia{e ����5KЦ�_���J8�pʋ��p�-ခtN�n��<
��C#���V9`J�r�0
�;n�	oh~`O�1(�#v�b�J9�'��wF�/M�&;�����*�Z�]��Qa��˭�"
^_.%Ռ���K��$X8K�$	��u�Sd|yw!i��f��%ݯ���5�����:RY�D��p�ۮ��2RŻ��|��7]��.��$����$}�TsD$����l.�S��/Cꚷ[���|D��BqP��o`�m⠒�S2��7�Ak�M��*�;�!�+��F�j�r��Λ̢I ]#וLR�+�㥌L��on���dՆ�% �N09'�
r+�q	�qiN������UgôUɉ)�@�=���l���:���-)����U�d���XRI�<X�k��sKl�Q�YA��*w����j�+�jBo�P�;���:���R�a�>�no�WOo	9�m�S����j���W�<x2HW��I%E��I /5?�H¹%H����"��Ώ,G+�D9Z�< ��nUHw����>��Ǯ/���G��hm�$;Z�?S��h��r:�A��I%5�#	䥪I8��a�~*�ZR�Tϥ�0��[ɥ2W�wA��\Z��z�\��LS�4ܭ�����j�-�����-�̇�Z�Gz�qU�@���$���$��:H��5Ͼ��7ˣ��H��j\	P�iQ�T��'���8��Ӑ���ǻ�*6�j�ki�����:`]I|Ȼ�M�M8��m�uH�-���T���p�K��'·�C���1�q/D֐1Yj�3!.-N&���-L�ͱEs�c�"��!���+T�7�J���6�>>���)î{�cH�0K��4�_<���:0�u��3�|�� ���9�zf�]O�e�6��ȩa��V)�y���QB5���j��P� �3�Eg2HW[x&��3	�
���s��wT!�1`�� S_�{$Pq�#Q*��	qǛ>q�/?���� ����_����N���?�u�����t[?V��n��!���?�T�q8��&���!�������y�L����L�x��dJv�+}'S���7�ɧm�y[?����~cZjk$V��,�0�q�L����I�dC�ǁ2%��)�JWB��C/E�!��K�S�н��~*���F�#aʣ�m�������N�䄸$�?���>8��"����ċ��Ð?D���F
=�&�v�FD�sJ�^�DĽ��lVi�:t�ɨv�� H�Ia�	���:0
;����q)�JI�z�\4m�Q,q����J
���ӕVB55��ja�D��귦$����d���:'��[�r��-��쯮T��8@���U0���� w�骔�G~�)�dL:M+�A�-cW��jRՇ�Z�� ��SM�z��d�"u�䏗���J���k��*^��E )W�ŀ�U�%�(T�� �T;����0��Ew!D��e o)�'T�<�xx����ju�X2H�.�ZRI�<X�KMWՒpnɃm6�bO�&��"59�j�Zy0F�<X �Nԋ���Ad���v���Iy��`��>���W��jR燇ZY<� �+u����W���@��O&)RO�x�֒'��R�dՖ� )<�L�:�P�pgh�!�A}02]>�����w�C(�;�6��ỷ,�j��S�Y>/��C8�Zw�#�i����yX#	��a�$�k���_uR䆡��. 6'P�1sH�T3�&ĥ]~�|`O�~EZ��M<F��_�%x��_��<;M���~�ٓ9���z�52���2�!�*[�<��R#���:tM8p@���h��2��b��Z*=��0l@ܺ��۴�ln�76�⢲�sUmO�^�ŕ���kre"��і*��R��o�d'h�.���W�~8奦�)<�����Ӿ��kɵ����{���(�rm������G��4���(h%J�YB���X�\/v�]-����-���[��̏Z���KU�B��sv��ؘ�yEl���nY7��-YH�y̍*�.DټO��X�\����j�d�&oɚI%E���Z�ᙄs�ʇ �*��������z��w��
����a�*O�0�Ilh&Pqk�)Q*[+�	q�'�r>�Ʒc�p�OmU,��X=�{Z+�XWS���іf%�'Xok����V�K�r��S9tM9�:�\9�:�a����Yl��8l�J�K�;�ЉK���=5�}����BcQYʎD��(���Ք���hK�C)�z�C��)�v�R=��bMy=t�?�q����c.3w�@fFC*��T�:Ç����%�x���V2��
z�&��p4>Å�&��;S��	���K�[.�W(.�[��f!�8<q!�2:q!�.e,��<�&�� C���k��~�+��#;�'4q!)�12��en!�}`b��b!�Z_�%�4��%�u�)	�F���p.��-��h� )9_�ה��G�S���Vf�/��Kg��>籟A.ŤP7�ܵBp�T[���Ӥma��H�G��rq�T[Ɩ$�t��HRI�<H�k��K�-y��:�C�,�Tw��V�S+��(�	 \�t��A��[!�3�a�
N:����_G�g)%T�<(?<���a���\���^�TR$��Z�I8��A7U��5 R���L�<,F�<,�%�    ��>�Z!l��7ܧ�
yX
��Z	�$��je��P�6Е�e���"y�$�גK¹%�و�wG�	@�g����[�(���b��I�y~PO�b\�x��T�^�n)|$T�q�|x�����Έ��I ]Ń'���'��{dy��-u�MF����;@~Q���~�0���� �6
���h��|����3���``��CHB5���Rb���I ]��$E��/5���|sIll2�����z���Tl0�����Nk�vQ��6�&b�6���o.6��%�P=�����IEz=�d����p0S����ÙZ�����*�U<���^[v�M�b[+�g��/��j���p,S+}K8��zKh��т�N������@��Ҁ�ͷ��}�[��'F��jk��$ �.[+��ʷ�, ��(	�<���z�oi�y�\��*NA0X�*3�+��[2��>�?`�A1da.L[�#�g��Y�J��,o=�&�4�5M*)�My-y�$�k����Y�B�+a����8��=Q*<�V�A��=H�z�����b��fhT��L��6�6������G[�m�`�`��qH�k�Y�KU�ANy)�ؿ��A��C��O�G�7�$L�����U�+Q
�C�%ĝՈ��{PzsR����͕|rQ��H��;�r4����o>>�J��L�~U|��~U:��4���K�t(Ry)/��]�&H-�Uܕ�0�0�E�"}��of)bw��f�"i�n��B��V_RCx���QCf�#U�����իd��Q�+.mr�!����*Y�ҫ�PM�J���Zh�3v@���8��� I%E�JIy��$�[�,l�VG��H�����/Z�jX�e�5pI�A���os�&�%*�ɋ���ѣX%�6u��hO�aw4z'Y+	��谒I��a%��:X�͵IV $�ʋ�1�����a��EˍB�RYT�E�[˖^����b}�#̫��5�PS�(K��4u��Ƴ�g[�̫yg�zO1�8�Ҕ�5uz��G�C*�5��yH�Z?�0H˥�-H�\6'P�t��UZ�7�&ĥ{A���w�Ի����1p� [�ҁ��c��`~���7-��T=7-�7r�V�r�!uŪ�#�Q�*߼�苸��xsCZJe��!'�.��T>�7�{���k��
�����XD��u��!<�}�yI{�)ֻ&�I ]�sO&)j�x�k�G<	�V���˪(4_ �N�27��ʃ;0
�a�2�[�#?�����i68Di)yQ�<&�[��A	�c����Z��Vu�$���y$��C$��y$��R��dT�n��O�fu��c�T|0�����dՇ����v`��$&\H^������l�m���r@��z��QQ9�p@}������6�H�h���#5��3�����G.�m𹹨:�֧ �Z��V�F�6L����O��郆����Sj��� xG_�i$T�:�|x��[N����)	�K(��H(	�*�$�[��ϤV���T��M�Ԫ�0
�� ���$?�u�7Q�cs�5hִ�|+��q��PM�`��P+��P�SR��.q�$�"q���O��$�b*��!PEy�T�*v���J�~� �K������fl�&�2��E6�@�-�(������c�B5Q�c?(	�I��Ij�!(���!(�����g�R��4:E}2;{J���=�:��2���϶4)�W���)��6g6�4�*g6����<O��9p�"��gL��0թ h&P�3�(Q*���	q^%?]&�fg������?qzy��!�T��c��=;=�X�1���і���'X�r�!��x:��.U��S^����7���/L�j�v
��6�y ��1� ���M}x�l+��"��ś�
��5��4�����n��~�_�o�U�}��ޛf�I ]�aN&)js��K�ٻ$�ܚ���ju��ruPL�:��P4 pI���e�I�K0;%�J�&�[�aQB5����^=����UVH��K_�$E값?^�1�+��:���z�:� �:�˱wU|�ƱvUy�Ʊp���X�7k�"MF�D�y�J���s�Ʊ��v�Ʊ��s�Ʊ��{��#�i��#��H"	䵊�H¹6Г�R�Oӓ��R����:i+�z��V��	S���8�8�����8���1�ҵ���8����!��2�XWš��l9�c����D����-�R<�t!�T��\0���"�(|���LY�A}�$!�s��l.�?g.��`�jz�xx��{�: �yM�5��K�I%E����Z����s�K��� ���]�Ů�@�@�'&�<p+�+I�a��̰�F�Ui�*h��
z�0�p�63L ����]�<kݽ���$1�J���#	�J��I8�voH�X��~�H�<���xo?vag�{�A�K��#?�盌k�.�c.��V�|��-�7L	�$�O�RX �Y='�t-�pRI�<p�K�X��pn�o6��į����	�ZyF�<�Nd����gr6���ƜR���nQY	�ӿ���Z��jUIiھ
M&)RM�x��Rh��-u�MFZ<z���5�PL�:�F�:��R�ԓ:��V���v!֥`��nQ�5�gC7�|x���X��̯XI ]값I��a%�Vki%��R��ɨ��H,H�?3V �Vl �Pl�R���X{�<�����;�������_�ç����~�����鶞ܰ�N�~XHWgɒI����?^�v��[�����Z� �K�P��΀�U`��+ �$5����>�-m���h��l�
"~��-���PM����T+[K1 �'���ޏH*)��Hy��C$�ܒ��/>��ұ R>xL�<��P" p��Ӎ����wΟ/Q�8o���<).�����-��x��3|���2�mV�a��	&f�yi�%����D�R2�S�<���F賸���(P�׍)`j��-�.2 �Clt���Z�����Q�j�A�ޣ<�)�m�|x���o�	X�Z�I%]yo��S����J^j�SI=״B��˵B	(�Z��b�PH�V��V����#�3|���&[#%�C�=Z��X]Z���`K�b`�j�J*i�Ml��*�XI%/�+��^z�{�Z,TW�%Nq��z�T&Jk$�%��5����r��%�����x�F���}�-�E�����G��f|���Q���k]�S�n�C.U	8�p�k醮�A��cM��F�-�5)a�g���8gJ�TF�LM�;���>�Ū��ͱ�嫖g�����q+�N���[��6����6�2R�sD��N��y��N1�J���s�Ã-�jMV�F��L.i�jQ�JU�II%��1(��Z���W��OL˽~/�p��B@*�B&n��P~X�E	/�0����x�V�$V�V�><�R���^����6���*��d��
M湦k�Wu�9�Ju��1�k�r�Tj�
 ��
�a}`�����:Lf�]�&�[���(���r��psc	�Zo�K.i�@X�J�XXR�K-���TR�5�د�nŁ.s� J�X�N�Xlm��+���������Bަ�,�M���r�FW��T���N�����3k��X�b��%m�HR��H*y��"�z��El^��w�(�r�6��^�v�J��2�^J��螟Շ�j	'vws\�-�
� ���̉�#m�����1/%v���ޜ���L�%s$�I��$�%s$�ܒ��դr��ĮQ�g�a�
�Z��( )Ԋ�?@�TW�����t{��d�B�c�^M؆f@߰ѿ�$��:Ps��S�4�͹���R|�O��J1�H���7\2J�R��S
J޹�4�Ĩ�U������S�$ �T
R �R
Z�a=(�-,���Z婞d �*�����xx��Z�X���I%mZ��)UZ��$/���sM+x�Ҭ�V`<��'`��� �R��K.�)#?��3�>�&��)U�|�G*����~x��.��	����d��͛    )I)UR!�$��3%��TȦ��uͩ(�K7S'p��B	 �Z������7j�J~XOeE&��m����*�Px�V�J���B�Ṗj�:��ҊO��~]+4����X�)UZ��I^��X�<״bmZ��특(�lWTi�R�k�Z �Ԋe@8Z���Q+�����-V0O3�Z2������f����+l><�����z�
K*il[rJ�VX2ɋ���sM+l��*oA��|oZ �X+| �R+|�ָ�)?����b]:HG�V8�g��XM0���V�e�/`u��o̤�6���*��d��Z����sM+b�R�r�ܕSX�V�X+B R���!�Z]����\Ao�����F��j�bn��9|��<���P�r���X���䒮�mI*EbA#���bs�	_�?�Q} k�,��o��0�ԊH�X� \*,h���>X���j���,\V��6�	���MJ��ʂ&?<��.MV��frI�Ăf�J�X̤����sM,����@���c�g�X,� .]� x��L����6XW��V�%U�m�_�);����$����	~��n%�tu���S����I^k�M��s-�ޑ|��Q�/p�W'�$L�K]�8:�W�TF�%ĥ�qu?��M�!���AZr�K=�����I�4�㳭��hP��ʖ�a�.����ī�ΏC*��\���e��J�	����-Oӑ��[�ts�P��)%�/�]4�2���mSI�B�6�"�5�2�(���}ɰM�VT��r�P��p�H�ԕ�8�5#y�.�m��"y0S��Wu��a���[�^�t��!���uH�t�G�n�G���pو�;��KE\ڢ�	q�Gi6��d�^æLs1����Ki6���3��%Ə϶�a&	�:�2;��5�2;�R%vH�J>�CB���	��G��(a�]sN�@���$�R(�Z\�]fvZ�� j�u��5��q�r8Y����8�kj:�㣭q9�k�y���mR��.E��|8嵊��Aה]<�r�@[N�����6\{��Jמ�7�5g��Ƚ�\o�0s^¤Ze����YO��ٛ�s]{6��ۮK��?y�>��|&Pq�����f3F�2�u�1���t���_8T��쯭�Ԫr���S�c^���<��^�'4�xX�z���>�җ�dq襪܏C*��c�8$t�S�c��9��5s�_��f9�/}i�����)?ؓt9��.3ʤ#"�[�>�k����϶R:|P�����J�9�ǡ��zRy-��qH�Z�?�$�����)2��*���H��z΄��)�t>��z+G�]�1/#�R���'z�rL9`M{s>��і*�\	�jO�yH����yإJ9�����k����r���0��*gK�b�`O�J��H�[�!�|`����f�V,\�&�{:�B��S,��lK�C$�Z:\����K�t�!�+:��5�H��Wo^{8�A:�Ԕ��QSJ�:M�[w(�My�꠷S)h�������N�&
\���Qگ�mS��:�M��e6�����C/U�a�T^���vH�t�謹z��a��Q��	T,�k��cܯ�֔����=hy3[>���iU~��"��g��a�ѧp ��5�g# :��U~H��^z~]9��K�r��+:�p�%� ��Yp���C	S��@�@�ʱQ$Q�cChB��Wm�u>���q�?�aC�����cұ����L:6^<>��LF�u*�=��Tsl��.5ʱ��)/5�ؿ��AהC@P�\9d%L��|Y+�x�T*�DB�R�{�W!���8T�q	�"����StrF�mi�**��]�gӦmұ��MUѱ΢�KI��]�Ŝk���+Gy�*�Q��pKu���h,����-�a5Kǆ_�{:�:���Ur�����/Y��_��e88l��6�K��cV�c��K�t�8��Z����nIG���T=$����U#��V'���Z�����>'����f��U:j�U}&zˌ<`B־฀�X�Ǖ۱6��<��5�yإJ9��S�y8�Z�J%k��N��&Y['TO�����G�R34B�f>�k�,4ܪ�I����.g	9� *�.J��N�ℸ�B����y[�#mGVѩwBR&�;�YB^f�uі=>��!��땎uX���%h�J:��KuB	I���oI����:� rWK�ar�J�"���j� G-��3�Qom%Ef;�Y�dS�S�_���=��)��W6g��6�͙먽QA�U����T��|H奊���:$t��@H-��HxԯlE���~�V��s�4-�����������74��`1��*��.���c�5���.������x�갪�M�OO}]7�PJ۠N�T�Jy1ݐCA�t;�����ҮrCɁ`�[$ ki�z �W���B��9l}�̪c�����EM��	�Sr X�O:�����KN<�9m=��J[ɡ�^��C���t�CBפc�����R��NG �A����Z��HtP���@ԃ>����\[/F8�t�4��x<��M�k���K_�<�����@ȃJ�r�!�����.U�a�S^j��W���)�!t�+�t�7���X82ա4|-2��Z�Zd�Ç�kB[UHxj�p@E�Zd�j�L|X}��j�U��́~8��[�\������p��[s���Vx�0�0�ˬ#m�ŷ6�J�B�H�]slx?ؓ�+h�J�������5�T�s+�FZ�K:6�||��5���J:>�e�Uט�U�,\cz)�p�yH奤c��	]���� ����)�96'N�r丿r�uChB\�Aix�eMj��q-n�i�H�i�G����u)G��{vYqJ.�:G���J��Ȑ�.U�!�T^L9��5���g�'��s�,�*��fK�6�4Ag``��%�Lrֳ��H�X�\�D�ԇ��7�Ly6|�t`���~���	���ڰ?�(-w�#IEk���PQ�u�9�����:���¼b�9����hD̊6:ᑫ��� �#F��j��1�����C/U�k��K�R���е׵\��bG�񄩾ϱ�"��_װ���96�L�;�U������5UcZ���;� Sojc_L��sl<}|��[�b��-b:��dh�x�]������r��kʁ�5�[İ��Q^tl K�b倵�����׸dh���zY��"�� �D�B�VS��������x��hK�>�5Z���J�r�a�*���)/�z8�r�湨��osQC��檎�g��h�k�k�x6���CeR4��<�b6x$x�p������B��d�U��W8�pJ�p`qb�,4o0O�֨��� ���gg�W%�v�x/6��3��%��U�мaV�T�X�@�@�R<��\��"�Ҵv�����^. ���M�`^��Z<��\=���X��g�����3�q��Ž>�I�T��c�a�"�t8嵔c��k���*_.��3-�o��lgV/��f�.��c�5����rA���準�.�n,��}6���t����m��-�Q��mZo�1��J�r�a�*���)/�z8�^<�d<l�u�L�8��V�TF~�%ĭ���{_0�o�~�Cr��,I��H�J7�+����-M�vJ���o?a�m�"��KU�Ny-J7?t��=O�6�L<���L<o���L<���������=���q�(_'�3+�j>���k?�ks`�6��H��	S�|X�Xőh4���*#�6�L������o�c1c=G���"�!���dZ5	��ƒ��wLX�$;�'m����a���ez���8��Z�s��}�2���%�3��[�&�zV���F�D��>#�gݺ�������6ޜCH�[x�*��Y=o�X�tL�����/4�۳�����J��ԍw襪�<���t�CBפ�k�K�3�j�D���<͍B�R)�Y��47���Y:0��k�fިR��դ��Y�O9�۳��i�(r{�jV>�ҦrإJ9�pʋ)�������    ����^~��fF�T�J�L�)ݕ����#�{�w��m��x��>�W��%s}�I92ŧoWzflOӮ��؞�)�\�Tڔcv�R�u8�Ŕ�]S,�(�9�M`�<Nsb�ϪW�'���tUzbs��5��J�=�J�;)ciB�2{���ĺ��&��r���JO��YӮ��>�=�J������Tڔ��T)�Ry1��CBה#��A�3� �`�ϙu�̑i��a��g��xNb�7�A<|�9�%5G���g����g���1���g���wD>�J�sn�a�*��)�定q8�r�8`�)'�}��Qt��0�����JS6�~�[E�`�)�m����9X�(v�k·H�`�&������(I� ���z��U � Z�W��桗"�yH�A:>�i}V:h�%X%@˪�� 8l֏�	)�X:ra���x���E�����ĝ�ג1�k��0� �G:����~!���
����6{c!��tUD�^�����V����5�s�������_E����Bʅ��t�-L�I�/�3��v�Ჿ�nU�B(��ܹ�`~�����~��F:zaһDLzX���$顗*��C*/&zH�t -ê��o��0����Us	n1+5�|b�n�:2{6�ƛI�,���:h�D�):�$�>c.�;fM�\�w̚����t�iv�R�u8�Ŕc��0���p���s)�b�Yt�n��,:J��߲����oT�t�=g��XL�?�#fP�#JsYS�����#�cMYt��1���RikW�a�*���)�5$'?tM9�tV��A�"�!9�j�_��σ�Y���5�:��i�~�-M�K�`7<���˸�K)�3�TzV;�)Ï��i}~��M{��Wr\���i�\D�a��H�C/URȇT^L
��5)��δ:i�`�3m(���V��i+�T�4��4.NG�s�
��ؚFA��bRv���-�=����6��4�5E�p��z��o��*]����K�t�8��Z����-�`��|TW[�7��e�߼z��a���F���ǭ�������Su����_�����7݂ex�|�I�p�~����`��|�VL�Uڤ��TIRy1�CBפ���r�@q�~���y�՘a�R�1����t���V���]q���͖�W�r�G8��X�p�
���5�+��f��)m�*�h[u�aE�Js����:��m{�Q�7Z3�C��$��胅nk�Q"�o[_�aaۚr��֜�Dq&ymF�8#��uNy1�O����;�i�KQ���9��N���֖�ap�i��v� �9����s�� ٪Z�9ۍ��Ŝ�F�rp�n��08G��i̇T��>̇]��|8嵔��p�5��ʅS�t��C��ТƲ�W�a0v%�4��%�K���=�Q9�>���0,$����˪V��N=���N}ұ��-�4`u��0>�g�u�X�Uڤcz���uH�Ťc�%����3u����V���#(��z�YPCy�J��z�4�4��&��ʟW�����aJaVv0BPqy�F����fAE�M͂�ʛ��	R�R��.E�!|8�r�X�ԯJ.���^l��g6x����� 0v�a��\��+��B/]A`l��ƺ���@
����X���B��޴���;��Ʀ�7m 06�y��J���T)�Ny1���Aה+�^����9��r��o `��k7`pp�Uh1����pl������UE*|S�w- ��0=�������7:��J�r�a�*��)/�q8�r�-�\9`�U��������(�*��FG1��|]k�a;�?��Z��U��&(����):J���M9{y�>:�$�\���W9dRikэ�.U-�q8嵔C��k�,��U+�`Cϭ^9$W񪣣$7�J��$w��r�r�st���ܒ���C|�wro�):JrK�1:�ӟ俦���>:���\˳f嘇T���B�]������r��k���{u��8���%7�X$�J�X$W���r�n���|cBAj4tT��%���I:rK�q�����ż�k,5EG.�y�t�a��v�ȡ�*�C*��=$tM:`�Q^t�	����ꨎ�X��4.C`��k��n��8.��$u���)*P8�����w}q�u4�e��1��cR��'�K�r�C*/Vt�CB���v�����*�����H�Z����P:,fB\2WY������Bn�e���V%�[p���~W9,�5��,���V�+�Z3�,�t:,�)���ת9|���_�p@���������@����(�����
�����>��>�Z����pU���߹V#c-fy{�e�&��Gu0����`?|�[7]F�x����n:ḧQ6���~�߱�+0���D�E>���$=,��|�X�pJ����T��v(��Jd;tM8�[_�$i���u�t�W�OI��Kߧ$��qm.�>���)�?�F�s�1��V)GZ�2�$�}A���A5�i�o�?�ҦqإJ9�p�k�����)��A�%��A�����AS|�f	��Aה>�x��y����`��IW�������1�A�E��#�3�J�Cᑏ� �U�:�I��C*�Ut�8$tK:��(��)��!��
[|T����(��)��qm.�0���\�H�BJ����<�P&�h
	RX�C��&�x?�+�t(L�!��:��I�<�R%�ʋI�&�N����<'R���N=P��4�@a��5�S8��9���	KJf"6ʎ�+L�єz���G_��#M�
�|<�|��]9�J�r�a�*��)/�r8�r���ʋ��X�r`rջ���M��*F6a���(f9�����������5�J��y�iyU1�	��q�iyU1�	���Vi�?�R%~H�Ť#	]���+�o iڒG��JӁ\���i=.]A�t_�ث�?�p���k�����WF
�#��M{5m�}+H�����v���5�tI��^��c�C*�%k�%k���/�������@�ұ�'J�t�	q���/��U:��{�ZS\g�5]�T��9�gH�@��j��"�$���ƅ���8W8�҈s�i��[F�͓�0�|�����5xT\�V�C���s��x�8��*Ν�����Ν�Z��t�o+v����p�kY�]������b�8�U���dR�S��cR�S�ar-�V3��9��o�N&��D��c��'�I92�/�V3��}Pm�rd��jV=��eN�uإJ9�!�S�uH�rd���p��kM����q��뽥q������57~��i�,��x����5q�mসA���Ǹ�Ң#�}���HG��z�9��@:����_�;�R%vH�:�j���ul���m�VQ�q��1Y���0Y���*y-ad���y#V�l��J��1�:�i�����J3��Fleh�J�d�F�J�d��D�R�*:֤��o*���qF�vJ��n�yym��y&zU��y�x���k���5�
R��g�H��Qբ�@��O�(��O�����|`tU�U��"�fŬj�&��AJ�����g[*�	��t|�����A�Uڤ��TIRy1��CBפC�5�r�O���%�@�ҡ#Q*�CgB\�8.��=H�m�ܲ�1�̪*����TXS�q�>>�ҲQW��Ϊ�R�:����K�r��ךU�u8�r,��7��;�Zi�����*V�%�R�K�Rv�Z�|`O+͋��M�6@�ʱ,ѿ�o����vՊ�G[Zs�H�֕�e�T�j;�R�v8��j;tM9U�[��~��,����<Q*�C�p��A�H��ޣ� ��l�>Ք���l���X��թb��Y�*�؁�K�Ɖ�x�Q�9Ul��M[��YT犪,c��Ԕ�)����KslQͦs?����|���a���硗"��yH�^�|�&��Uݼu�tgË�c�;�_���Y�"���u+�1��ϛ�6yY��U^T�;���i��1ĝ}�    ���lZ4p�o����w:��U�;v�R:��ZE���kʁ���#�i��[�pwV��9���4U̙��=:��}���a�ݹ*��Y�G9x�����=>���[Ξ`���C*m�!�]��C��X�!���)������I��5M�b吕(��'ϼ���p���5��\#����*È��3��[��|�P+�Yd�����Ǚ�<�_�}�����í��L�+���2�:?>�R+�K������x����FX~F}���s}z��g4xm�� (+7��H�zi�VW�+sSK�ɕ��v--.�T?�&iP,�9��HGƬZ��A���eT�LQ}_N���V��6gT�8��UN�8�R$6��V9i��-鰌_-/'-�T���epju9i��ZZNZ&�����2J����o�U#<x�m��˘զ��Pվz�2F�)��r��{gX6��I�<�R%�ʋI�<$tM:r<ʥ#���A:r��:��r��4��r}�Z��^�s��|Sqr�I0@TU����&���蓎\�n�<�\�f��*m�A�^������t�!�kC,,���o@5�O,��&Bp��>Bp,��&Bp���fBpl��>Bp,��2Bp��W#ǲټH�;��SY�u3��3�>��X$��SY�]1��P�!��SY�6�&�0�,U�y��2j����/��KY��6���U��aS�,�C*m��]������r��kʁ7�r��\�f�
�T��XS�,�P��5�����/eͷ%�$^�]j�Ev;F��G:0�X�t`�M�c;k�Cl����Uڤ#�TIGRy1�CBפn*��̵��0n���L��4�30��k���a5=�v�[�ʜLB1g��^`�Mԣ�A6Q�r���>��D93l�^�qH�K9bv)R���S^K9b�������k����k�����k�{��k��z�{�j�s����{�k�M��wH�t�ڻ�\{�\r_���G���_+�E��k��2�J׵��C/U�A�T^k�;��%�����KǆY	S-��T:6�'J�tl�H�;��3��縩#¸!f4}�;X�Kǆ�	�!�X�t��z�'nj�I�uJ�=�Ҵ����H������t���!�K�!k?/���	S-(�T:d�(uұ!fB�Y�\����p	��L�?Ʈl5��/�w(���W�����ma��v�:�U��^	�T(�������/�R�*��|ȷ��@P��I� S��	T�$�R��	qK9h��I9p.֘��eV��nx�G:�X�tP<>�R���`����Uڤ��TIRy1��CB�F�Ǡ�e�5�zgn`!���,��@Y r��ڐY��|���^_� s�j�+��HM����]ұ���ٖNɑ�H�ʂ�@:�A����:��Kը����tP�!�kҁ G���#�����F z�ʧ�H^��)92�ޔa��<%�7�P%�9��_��F�&�@*#�](�0��!y�r ���/��Ri�W�a�*���)/�z8�r�D+��g��kT	�R�]��V*����J��i���UzNBE�J�EZ�bTM:6�L����U�Bl���(Ԓۖ�-yj>��J�r��.Uʱ���r�CBה���/Q�Z�ؗ���P��#㫣P#��K�P#7�ֈ\������%��w�����5z��G�\��B��g[���5��(԰�*])��^������t�!�kґ��^}�)ra���En�WG�F.ƗF�F����C������*}����=Ye��\�oJB�\��KB�ܒoJB�ܒ�W}������Tڊ�8�R�q8�Ŕ#]S�܁(OB�\���AG�B۫x��C��jC̄�c�B:���>��0�t��F�@.X����7�_�`r�z�c���-�Wm��`��{6�a�&��x�^�v:�8��R��c����z�>�0�0����	T,X������	q�_������F&˦1�0��WmxK���c�����sO�!��so��=��&t�J:�ʋI�&د������^:�4�@���1W&n�H�[��1�|S��f�jE�?�G:�BX�t�v�=ɇL�W:��J�t�*��C*/&|H�t����ɇ<`@f*��@3���Fc�L>�����c~N>$HG��Щa�E��	�#p!s[��Ƴ�g[*�so��=��&z�J:��ʋI��&0,s�>�����ϯ�՘��7�J�J�ɘo�nx?�s	��%��C��8��g�|�ܶ����-�����cVi��u�J:�!���uH�t���R.� �44��5���CX��6�&c�k��1?Gηc�4o�F�Yѕ��	�#�!��I���>����G��1K�t�a�6�C/U�a�T^L:���5�c��W�7%L}��1��oI�J�ɘo��o�u>����~�Q���{3�j�r���=���X�r�x�=y�<`<fmV�8�ҦqإJ9�pʋ)G���3^�[<�#�V�g�r�4��oJ�B����.��w�~s�zZȞw���h�
�k���ltM��[���K�ڔc�y��*yI�j�yl�nu ��6)Ǥ�.Uq�t8�����p�5�HיW��f�Ⱦ�\Eʑ~��}����R9�(��U3d���CD&B��������.�Ur��%k[�x���Vnu�4�y�>��t�i�z��>��ZE��CBפ#]g����|�1�/V�8�.V�8�(����O��f�H��B����b��$C���U3�e�ӯ��%�[�i{�XSu�y�7o}�Vi�[�x�^��C���t�!�k�י��IǄ�LF�5w�/&��v1)]�0�ɸ�K>� ����Oˆ���k.+*��2iZ����B���5�yL�/�T0���B৿��.�J[�jz���uH�Ťc�&p��,���d64�����	���.N�d^;-��_�L�����}�	s�F�QX� �Ut�;&M���1��y���J�r�a�*��)/�v8�r���ʕ#]dVo��i+N��(�(�ʑ>1��i �85W'�V�]u���E�ͤ#�e�4%O+�cjn�Q�4����n0O��Q�Vi;J�^��#��ؔ<	]��΄ʥ62�����	T,H���U�PJ�kҁ�Jy^%���U��L.���DXJ�*9!�R�V�	��ҴJN��:hV�:hz)��T^��qH�t.E����k)R߯"X��z��4��}@��X�t���o,k-$��ۜE�\�-Y��	&d��$؎��6`�p�v,�,�pJ�p�C.U�1���pС�k����oȢ�G�GzT,0K�6 nF�p�{,�ۀ�M�:���g�3�k �#p!K�: B��m�t�v,��e�|	�%|�J:��ʋI�&�+�*��eշ���P�t�h,�����t�{,���M\�+%���fઢ�d�,�(��K9�:��u@���W9�J�r�a�*��)/�r8�r��,V��A�KÌ��h\=#�����)Ʒ.�nx9ؓr�cWsnnG�UU͑�dkR�� ���x��h+}��������~]9��
����u�X�]��cNy1�X���)G����)���Ps�ϸz���f\�HNi0��HN�<~^$��-×h�0�UUs�)�i��҂��G^i��4���`�`�ˀd�T�j;�R�v8�A9>�?�v8�r�[9ʻUi?��	9|�R�GN�K�9�`,q�[�|xXv�XS��$ĐWH\�ҴGN� K�9�t,���K|�ӱDoz�a�._.���*��C*/Vt�!�[����i�aY��LG���a��=r�WLK��.1���7�q�m������òc����q�jFΰ�i�9�H�}�e�1m:,�����]��yX����y�H:xRy-��yH�t�t���.2����a���[LKW:>1�[$�>�t����c�
tzb��s�>�t|������^��7������V��3�-��3g��Y|���ŭ�3g?oĮ�:c��Ȩ�g�r��{6b)W�Ϝ�6msg���    ��ʛsg��_��a��ʛǡ��קqH�6by���$x*Z�#�c꬗�NBu�d#�4�G��p-�G���Ó�N��5w.I��6^K�	#�ɻ�J�?����"�Gem������Rd�A��Ѻ�����EL_�4�z:4nz:4��{x(�иV��О�R��)/$thOQy:��]B���?���g��vo�4cgʊ)<t�;F�9y�k��ǥ������=��;�3���
r�a#���ٞy���=�g�ưJ����RT}楺pȁE��ԝ��S�]#�B�؇uk1�O�1�L1�I�q���� �lܧ�����,t���t��*t�|�l���j�a�*Un�+R$���H�0$�m��De�[^ك����Q�)�Q�?�|���`������m��SAH�؏U���Q��l��X%�m��c�x�<qΔ���MX|�H�=V	�&u�@d۾�8^D�M�ѡ�*U�4є:4E�a���k�@���|�����G��	B�Ʈ��AS��@:ۮU�	b��Y�7^.�{���}r �mZD����Oٶ�>Ad۴���RFKua��RTFO�F�B�K��m3>9�6v� �m�>A:ۮu�	b����w��{�9j�}��""�fE�@���:��m+��D�͊�1ST��1S]X�)*#�L�Fd����'o�\W����@Hۨ|�x�]������o��o=獶��u���L�1��p[]� �mE|�Ķy19V�J9V��+5�a�X�A�ȁ�����	��6�g���]�'���O�.�k|�ؑ}V��K��K�jʪ�$��hK� du|���}W�q�����bp�Ԕ*phKqa�[jʳ��-5�Z&'�ދ�ɉ����C#��n��hS�4���Z$'2۟|�e�H7�8��9\X'�8�*J�Dx{��C#������/��	�Xq��Ԕ2p�8Fj��9�t0�9�FI� �ƿ�RXŜ]��p�9�FI��v�ȡ0��g"G_G!摾}�u�P�ʼ&�s��VE�Ƽ�EIa��D���RFMua�CSSv��Ԡk����5 0�y�9`sz |b�m�E������"8���@�xm��Zm��u���5䀋ê�ۘW� �6流�34E�����"���<��"t��9�Oa �Qp�����Q̩�{
���ѿ��Ώ��������ր��c�t�,r�W棈p�y]Pa��  ��ۘ�ST��1S]X䘩)#�L�F��\�䀁�rZ � �Adr�(�� ��"�r��
�1��K�n�Ys?��,r�V�E9@����s�?�����ۘ� )r�m��r�?���{r��2r�T9Vj��ȱR���~3�� 2Wz��c�0�95����� ��<�A��w�a�gx�s҂�
c�62�*�c^TǼ8�;U��}Dw�;E�Y谖"tǙӃ��󃀆�fg��Nz��Z�P��A@y�o9k�8l�;�|�P��V�h�P��uA@C��]�^���RE�.$rXOQy9z��5r���A@C��;G��e�� ��d�� ��_�ׂ���V_{��k!!�BL��5��d�KL�^�4���kכ�HU:����ߣc����1RT���"t�V�oƶ�o�}�E�%{3�EQ%u3�E������f��|��'=���]s_eQ`����dŢͲh3�Ew�]�b��R�d�4充MQyV��4E�:®L�[��g:�h�N�[���)r�񼆎����k#��{ǲ�j�SG��gхU���b������x��RU�.�,充KQyة�R���#���9ua@��ՎNc��#���S��qi3�����O�Ҷ��dk���_�.BG��w:�w�}��#|ǻ���SU���)/,tx����1S������^�xƌC���Ɠ$78�'5Hn��v������������bKWV���K���p!Ϻ ��w<�����|��x��ɱRT��s�Jua�c��<�+5�9`X�� ���<;?h�Ov���4��$��c<�H�[99`>����d|�n{Ғ�c�_��2�U���Y�$7��gq��v�J١c���бST����O����)B����'���k��9�pX�';I�pOj��a2��V��>��I򼯚c�n}��3���9jN�|��3_���Y%w�gq��[�JU �{�	�ST�u��"t��G��)�T��j<�Ir��xR����$w���g�����:�ֻ�h���c�k�Æ<�V�:|ǳ(J��Ϸ(���~O���Ru����"�HQy9F��5r�u6�Qr��l*?����Mv��a��(��(6�E���%���RXӽ��b�沩EgX�f]����EIr�wl�%�|��{rX�J9,ՅEKMy9,5�9`:��<��E6:�b��t��&5����ky@��l~�����Z�|�q�K�{��%/+٬�:�c�(�����̀�*e�U���B���<�"t0�Mz���"��̀n1v��#ZJߊ��e+Vg}j�:{-�#�ԵX[g�����F�+�(c��mN,�\��m�)ڋ屪�}/�y(��|�Ţ8<�Ԭ�$�K�JU��%��[IQyV��5E�o_��Xv�Wn���%G���{�����kŷt�aǏ��1��kt�����1�N�H��;~`��C�F7��o�~\+�zcz�_�cX��X�E���%�ɝa#�U����"�����"�LMy9fj�5rL���1W�aϠ���X-�0ɱz��E�5��}�ӥs��V��}Kb|:��*t,{�l��X�jѱRU�бR^X�X)*C�N��e���g��v]�4c���BD�l;F\B��-`_�U�n�*b�*�C��=���c��ê��"�M�Z�o_�Z��uv-�#�jc�i����"���,�`��w]���}��݈t��ݙo}e�+�X�+����C|�]{Ѝ,8�m2�݋ߗ�⦿MF���6��{o����|��9�o̘G��m[����w��d$�W:"���6�AGD�W1:v�J:v�;E�a��)B��Ypz7�G��ז>V !R�얾!njKߌ���n���ϖ��k�6:��6����1#�]�>#�]��7#�]��7#�]��7[�J:fKy!�c��g�c��[����X��>\(a�|"Ž�-}!�Em��o�v+�<�^�-}Y�1[R\�n�[�@�{��MD�W]M�D�{��Md�W+F�HU�rDΑ��B�HQy:F��5t �:�0c"ݽ
j�&��]�7�XԚ��T�����ox����@\`}��ɫM_竎ɖ>�`]XMD	���S�V�$�wMH
�^��nJ�JU�yJ����0tH��5t b��5}��5
Ё�Ů�( YԚ���5��� ���-/4?c����Mԅ�����r�U��>���G��Z��Y����RFKua��RSFK�F�,zK�D1�~�w�d�k�&
@��o��c�5r�d}���5��<��������f���g���� ���oz�h�g��,)&������S]X��Ԕ���S���5"��/�9c��&
@� 29��q�0Ɂ�3��#9������K�9p��ǫ�K���_�_s]2�u?L���l����3���a�T�������B�LQyV��\)B����r��	�r�V�	��b��M8���o�c�����L*0�Ϛ>{��/.]�}��sAg���5����{Kߏuῑ�����ǹ���xy19v�J��j���ȱSSv�ةA�����M��, �Ƌ]�7a4^�8���x�[�����38_~��;%=H�X�%�Y���e�X���<��=t����*����R�u��:VKQy֡c��[�X�+��~�X0 ��G�
�1;��hL����[/+�ǟy@yM���T��;�c�-yռ��0!��W����JW��W�K��)*U/k����1RSF��t��+    �F'�ǻ ӱ`4��Lǂ�xS3�ݮ������X����m,��;s���[�&��t,���w��s��x�brH�J9$ՅEIMy�uՒԠk����CT��&�aw�(�0�ܚ�5�0[3�ň;i@E�A���,8�����rS��lE��W�9%1��J�u���ɡ�<��ʇE	B�!��� �H��h5�?��?�n���L�ć|�́�˦��+��V!���a���ʦ��)��0������32+�3����9u�~N𦲝u]�<�_s�@zêȫ�0�B^e��MU){�Д֡CSTv��k�@�e���Ϙc؛�*���X�lj�c!���/w!���8k�s i��\�w�Y���]��X����D�q�-�۳���T�r������)�:s�?Uj�5p ��s[�,p ���_{��P��۪3e�&9M��V��o�_&2+�󶪿��6��kO��B�,���9�(ѱW���U�3�*����ǟ���1SU�:f�3E�a�)B�Ё��6::\��?s ���Y��|ʦf�)�n�.DV��fu��q�ѻ7��Y��ٕ]\H���( �i����߃c�� �o����S\X�ة)��کA����U�ξ��0o�o��povp�d��Q�{�����(`��O�s��{L�֍�D~�K�/Aǆy�e7<��;HAǆ�x{��j�T�*t��BB��)*�:s�"tm�d�������h(ǰ����,��E�QQR�y,�G�ř���h+��گ1�p��)z�1�ݣ���1��u���\�:$F�Z�:4`��s�v�*˖��F�`S��c��J���aE��QM���R�1V�<<+:�ch:$充IQyT0I�V�-t�݆��-��fm�DF�xLa�Cf���`��#��:��p�v�e����h#���!(:�cX:���D�����jO��R�Mya�CSTv���kvQ/��v���v���p�����rE!оi�������y~a�sk��-�-B���5:��6\�(	��pɹv���][��[�JٵkKua]����]���k/v�ؓM��҇]P���݁]���܁Z����a�Bǎ���CG[��ˇ�f� �wtA�"tD�C١����-��0�_�$U����B���<�"tQ���և�_S�#�ˮ`ۑѥV��H�kf���~V����j����9y�f�#�El;��lL�����w��-�`ۚ�RFMua�CSSu]u�T�A��I_�uՎ��.�	FF�}_�#�K��ڑνv_�#��y_寮�X��9{�B�;�U�U��}����#�[u]��]LKQ)#������);sxj�5r��Cڠ�]c����j�n1�L���$��qic���?�o���ݙ"��&]'#��}��p̖ê�q>��O�i�sİ�w���Ri�3Ņ����0p̔�k���'z��+��[���1���b
�ǈK�ֽF���$�t�a���0б$����Hg/�aU�X���R��cX-:V�J:V�+E�a��)B��Q�AGGdwG�;�b�~�Wls����{g����c�3l��=Tkm#�r���+"g��aUO䰎�S�BF0�����T�����R^H���RT��q�T)B�"��qХdt�1c��8�f�F:η�B�t��cĥ��l�?�t����s���4`�=�W��9�(��$o5lg�ư�HG�*e��)/,t������]C��M��@��	����1��������q������ұuk���S9�#o5��g�簢e39��]��!r�M�#E�����"���<����R���9��n�<c4ƐÀg��28#o���3bƈ;���?����s��c�*4p Eެ�ȁy{o�9�"o��p Eެ�\u�����CS\X��Ԕ�945�8��$�oi���l�c��b
��cĝ@�?��g�~�d���!F~�K��WEI3�aUg��ϖ��V{�T�2tX����3��]C�C�:>b����ϸ��6c��3�bĭCG8�?\�2�������SG��{ѩ#<��Ea�3o������^����R����Kya�c��<�����k���#ț�8�f"�#���0��cĝ0����ә����-�֖��gz��5��0!���<y�h���p�vW��)*U��m���ȱRSv�X�A���rgg:���[��j�D&�1�I�-1��������C��վL���1I.���5���9��c��ϖ���b�/���ߣc�� �U�o�q��<:��:���z�8���!_BGoP(�}Uoc�OF�ξ���w�}U�ø�[O���y_�_.sk�sm7��$ttؒ{�}U�	��2tt����}���k�����T=u����BGOQy:z��5t����q�3f�>:`4��8�v��;3xF�q���8`����Kܻ�N��^<�4�U���^���8����~����R�����B�HQy:$E�:�W�J?u��ܕ����4��8`�ѸS�����q�����޾s�pkc5emx�0&w-�:�V�B�ǽh�l��������CS^X�����CS�����n�S,��
N�wv���jܩ���q�k����:�����v׽P�O@��݊�r��ut�{Q���xܭ��R�Kya��RT�K���@��$AH�:F����|	�+�G0�o�c?W����Ь��Y֖RN#�Y��|&�W�3V#�������\�Z�հT�*tKy!�cX�ʳ�1,E�: �A/! �t �3�%$a�A-!�c�%��@g|��̗v�����"JZ�q�KL/9tdt�{		��1�E%$�!����)*e��T9<5�a䘩A�ȁ�Π?u$v����;���1��ԧ�����[��w�7���o ��o���oP�|����`�~���Z�*��_m��i:�ۋ�Qt@��՚u����D;`ۋkͺ��T�D�Lua]c�Ԕg�}�L����l�79��>�f��E�g��&8���P �?Շ�csC��|�q�'p�>kR���>븁l�YS��V�j�WJJ���Jmaqc��<�_����k�@v���ԟ1+��߉��H��[�z�^��f=�����3~��O����}��C������U�׎Dl]�Y�lQ�Y��*vN�T��#�Nya�c��<��]�Č��?E�����D�Č�+��fD���j�_0�����~V��k���Z���z���l͞�3l�*�D$`��i(�U#���_`���=�g^������<�j�Ԡk�@tv4��c ;Z�q�����ȼj5�@�u�k�;�`�g5M��e�j�=|6�⯁��Y7|��V�$`Gͺ�3Lc�/���ߣc����c����1RTv�)B�Ё����w��,���V����`W�d^��f �:�5�b�㳚F_�Df;�8�.V3�@B�L/y���bX9���4r  ;������T]WMua�CSS��Ԡk�����CtQ�4c�x��f��'�1n5)���E�{J�}��e�%��*��Qt_�W�Q�L3��3��i8��3����c���:v�;E�a�U;E�:$��ltH���*A�u�#����FDY���!��ψ��Đ�v�j��*A>vx8i��u��3o|w�x���m��ws����p�-J�£7�:ly�󇓡�-ow�����ʜ�+�5;ϭL�\;���1����,FA��a��pSj��{c�-*�`1�79���=�wG����2XQ��%�X-F]��\1�:���Jс�-1����<:P���Ё����o�������,�,@G�)�i	;5-#a�����pX|�e��{��e��\��BG�/f:�k1����a�����*e�)/,t����c�]CG�2a��5/�$t�������SP�2F�u��,���ˌW�M�l��f= 8  "I�/��2V�UG�0W�e$��q��2rH�����0rhjнG���Z�|4:�}�g���P�|/c�[>�=�-T,�ծ��
r����.�%��ԗD�#5�%��x-�%Q�����/��_X��s.a�؉�_�(
�c_�o@Q�X��\��q�T��o@���B���<���]C����VV�?%� h��w_�,%Tﾠ<J�5t�TJ�����Gs�a�X�Հ���3��G_�U�F���S�;	�,R�UVf�J9V��+5�a�X�A���:9`i�Q@d���D����d\#�	���o���7�殴�:�h�� � u�}A.A�����	A.AF19v�J9v��;5�a�U;5�9hzՄ � ���ף��&�G�VM(��n�>H�����UM�n�|֎�ERZ͙Ca��VF,�z�h�
��ڇm)*U�R]H�Ж��,rhO�E��(��K9Q~^X�����C�򍺣C��M��#��>�I����Yj��>�6�&+e5�ꊶ�k\��%�4�⾓Ir�U��c����c����1SSF��t�;�ЋQ.������#�r�=G�/���[�0Ļ�����]V������wC�[��Q�o�+F5$���Ր��]K���o���-9LS]H�0MQy9�R���9�a�s(�W"�wE�J���9+��S$�Dn�s(�W��m����>��s��&�,��^TEK�����'���]I�FYE�J�7��HQ);s�T֙c��<�\Gj�5r �%�6E�J���G��v�"f%�6E�J��E�J��d�9�t�f����Ċ������6E�J���+)nCPIU�:t�����!)*C��]CG��8q���m�<�z;��j��ƍ�_CG��|�ۅ�����9t�iśtĕNQ��ƕN]���m�w������bth�J:,充KQy:,E�:�z��"�(��ԁp���t�F��(R�2omcP��HG��6�Q�|�u�@Rf9�{������RɁ���訧����S]X��Ԕ���S���#r.���J��i��
�GG#�B��ш�\�;��Y���鯵��1��ٺ9�HG#ϲ��镺"��JQ��F^��HGw�J:v�;E�a��)B�Ё��6::�\�_"��PDT��@BE��dS��B�!����Ca�����������U�hQ��_ѺL�!��ߙ
:�m�谖�R�j)/,UKQy:��]s�"��CtE;���Hv1�6��bگ�2�n 8֔s�[t�v�eg|:�%ð*t�=��$t�=��$�SU��1R^X�)*C�H���Δ^Ab��(@cʮ 1�ŔZAbp�鸆X������|�cG�5��:���y�`/Ӣ��3��u��c��A�A�cZ�Ab��R�Iya�CRT�I��x���0�T
ЁWe�:��JMu��U��> �Lu�W[��9r�����b��J:�P�C< Z�0x �8�a��R�Oya��ST�O�����0��
ЁRe�:�*J�u�T�Z��P��߱����g}��/�F{�@�-�4�h]��P��߱Jg��bE�c6SU��1S^X�)*C�L��t�(=�a([�_b��_ep�);�a0�)5�a���X��C����C#�m#�AC�e�E耛L�b��~�:8�}L�c�RU�бS^X��)*C�N���Δ�0���X� %-c�X��a�배��k��w��˖�{LCh���/+ڏ��&��ux�Ǿstx�Ǌs�RU���-兄o)*�B���K����?����^��      9      x������ � �      7      x������ � �      =      x�E�a�$��w�
lcwy�?�~j�����d'i�,˦������=��c�ʝ�?����ڽ�M���h��J����m���g�ľ֭F��^c����d��1j+Fz��u-�߾2��7��{�;����b��e�m},y���+��9m�����ƋmL��s�����)���5���4ｚ���}�����a�9!�� WZ,���Yƣ��B�5Ӓ����F��s��Eg�}�F�n�������"����s*j���ò���\$�H��F���b��Χ��;��� !��^���4�qm�����f�*Ո�׃>�RpM����<�;pl���&yЅgA�}蘻/��%�ǅH}�Go������Q0��&Yz&��v	�j��}Bә�{���-v�GI��ĸ��Lسj�:4mDy�M"���s�4�8@#zdA���aădz��҉���l��[�oA{>�����X-�e�9���Ɂg#���ەseN4����?��`��Yp���P��QI�:P��$>!y�������!b��|�>���bܟQ�dH���ؚ� w�3�f����emN���C}2�&��JN��#�*���)Q��b�Qgϥ�N�
��]jǓ�%kKZc1v4^I��Aw��9`g	�;�zyrR�0q`�k�qen��7{��X��N�O��9�6�'�l4O�g�4�Kl�(�	@�FԹg6�u2�L	���R��D/:dq �7"K��u����܈��;]��8pI�f<�O�E�O+��_�]Ո�^2�"UA����U����G6xg}�p"��>t�0[|h�Z����bث\���#�1Ɇr"���j��Ic�Y�x#}����7�����6����J�I�N���4����!���9��W�6ό��hQ`�g��&�.:�ˡ|5B���X����$���
��`]����ţ�Y�@��'9����G��U�g�2�?>_��M�,�S�`��!��eY����ljs K��[�Y���[pR&�&	�VW^���S��������غ��F�g}����8�1�P���l?�&E���<�i6pX��j�&�=��lw?��d	��$��J{�(��� �H[��
,�@�=���o��'�'�]H�����ӽ	INm6�y� ������V�'
��(g/�㷋���\���bK�� �:┒�����q_����
�<�w��d 0�9�CI�7������թ�mQ��'��Qb9t��C��64ڝE,F��'�|�v&�7��7�=8�]Tb26o�y6(�*���<d����Z��ZM'�l\�:;܂�`�Q����gv��8	R?֠�=��7!H��sb���cN��K���b�f�L�ޮJ��ƴ����K.~Jd���D?�������Sճ���X���?�"���	d��M��c_��B���*���k~)�r�U�:�`�春?�d�`��-6u����<�R �1�D�'X�9vjR��_�)+օ�3�-�Aa���M��cX���AW�t>)�z��J1��*� @����?)������T�5�xb(�5h�y�29b�]wd�0п��8��<|�S�!YA��Bq��ӿ%!O��t�4�îe����uխ�eA�:+t�kH��p\0Ύ��L�c��6��сA+{.	�>�n����M>�J:��x�T����W?(��0�@�4�.�؞?#�(l�I&$��u6�}X���ҩ��Yv�����H�Pj�.�X���6�=�y���Mܲ�*�kI�������t��H?�EX��M��c�p��c�����fB��9���� �]c�vY�kT�D����"b�e�)�����]��V����&��OA��f��p����X�#�Y�1�Pω0�w�X*�:6�Cʒ�P�C��u"��D$�l"���X��4Kj������r�H7��Z�V��i�A������r4�̎W"���8&�o�����x+`�%��[� 0U���k��`p���Sկc�]��ҡ�qMh��yi�H���-5�$��	���X�͟��2�~B�O$��x��w9[r��0�̎�R�c5:.�a0M�+{�v����i���)�D�na-���<����y8�"��N������}�"� �tS@����$Z�.J �����mK��8y�����i�eG�0�]��~v}NVN���U�c�֩�X*9�#@/;�pIC4���P��:]�&�o@j�܎�"%8+$�v/D1d���K�T+�}麟�᭭xZ��ijtABkh ��*�¤� C*O���u%�&���J!e��� -
�	(Z��3m=Ŋ��]��C#�i�{�݇���9n�q��|�}�`zTer5m߾)D�tv�BaaF;{Et� ������&j��;6j,.�|s��&���vD����v35�����4PtPZ�sۏH @;��VcC�7�Ӯ]�(�!K�>NC����_�f�*�`ۙ�h0�PXj�:@�&���?�A���B�ԁ��%���Q%�G���4��%ޟ-U�O8��K��w+e�Tݓ�����5�[���pA]<���>\�����W����j�'f��]Ԋo��'�AB���d~������py9z�&~���t�~�T��)�kT����RE$�[-|�WS]m��Ԙ��n�t�|괚�ʤ=�@���!��*�X?���_n���rbb�Zz���ŕS�x���pkڦ�ޯ�o58���:���Xh�۰��vFn���o�C��1MI����M�7���]��p�T(]�*��&8��Ć9�4
��J*����S���i0I;���zq�=_C�������]�C0��g��,����l�ݯY�eu�p��Nj47�՜d
����G5 ��)4|��d�̀d;=n�(7�8���ҝjL�j����TAR\*�@��v�0���)h�W�xrm�T�5���� �$N"�5; `���1��FTCV�7b��6�Sg�i�ⴄ��?�~��ɀ�-Tв�t]�:
4�J�	��X��x]�@�<������I��\8Y,��S��q�r0�2(9���0:�[F�l�"i\|���@[MȞ�{�i`���x��p)�iG���>�"M�է̆�	��W�T]d5M�S��J?�����B$�j��_;M�vH�B�B�7qӯc&K����Oo���O?���8?�����l8yؑ���y3��7n���$���nU����6rLm̒�k���J�kIz'� �+y�I�:tw��av�&8��dk�����T�����x/�p�Y~b*����F���� �u�f�4Q��`�P�~�^Ǯ��ђ�+0��^�~�[�*��M�f��h|��9!d��;Yw�.��M�)������]TH�4r�$R^t��^�a�5�Z2��Mg��@��r���L�H���Vnc"%aP�k��T��?pr ܵ(=�L��W�eu�F�a�5��	L���i�UM��)�(�\���>�GuDJ��M�5���N��1��D��~��=�� ���P��i�'rhb@=���)&
�&um7��l;c�N>��W�i���#�3��Q�
9}�q!��9�BH�0�N_����Uyݧ�!0��D�f�C��=ئ#��ڐ�A?X���=}=D��J5�ݥ˫�8J-)��@��	v�������6�'?��u����_�Д��t���We~N�@�ɉ������5�2�Z��m-��:�,\�M��z#�?�`�&cC�ҵ��;�Ti��^o��Y-6�T��|$lҼ���ﰒ\�4k�aRU} �_OKop��b��P�����L3N��RW z��6�)n�t�:�mq�K犧����.� �L��ј�2����]�j���.p3��.@מ:{:׀��q�R5Lj���T�f\3�����T�uM�D͸�T\H��d�V���nh��5�s=s��/���[h��Z5x�Q�:pp,j�e�q�<|^s�g�͢�Q2�jܙ.�eo��BU����7��E��� X  wu%(��?S��v�g� t]����V�Bn4rmbP����ԗ]9�3J��F]&YV��zD?Wz�(������_�9��:-�����o��y��]rN.�5M_��D<]'�����/|Kw*����JdU��"�O�i��&���Zݥ0����8�t|�T��PB�:l�i��<P�&6ǿ�m�%?^��#�������E �X�9�M3�Ļ��54�yYU����։��% �]�M����%��k@����״�Ƀ�/�����5��;Yq��=t���{=�g��@�ì�|~8�G����C��`t�*6n���5טM	ǎ]ki`��Ew�ȯ��sO]s�)1"���)�b��u����˹Nkʄ��{�Y%�>�~�X�{��1� L�CS�X��� ���u��`�h��V�7�B-uҔ`1~�:o�6�L��W�!ug���֩���qeqw�B�,]�PV�]�/�T�s�nR�좇 �/�0���.,��Д	��r��d�	���+�lW��Jh�C��Br!���!լ���*����9H0j�}�	��)�M~��C�'m�|��7�-5]�%����Z�컂b      5      x�}\K�+1n�V�hC���,:A�N$��9�*׻��z�,A�(~?*��|��_��?�����=�����_����O�?�<ǘ���P���S糛Jӣ>ʣ>����Ͽ����LԿ����e��u�|��?�\����b��c6NUL�/��ֵ��VƔ�!���#�lV����x�u���^*&r�.I�1~�~�<��0ۆ�a���l6��y�-�)Ëԍ�"����Qˣ������a�*��ϛ�а���8{��8~��OoCX�
�[\͏	:���h]�X8p���N���S5ĤNU=�I��9q��p�*���ʧ�#�6�wن����"�5\Yņ�V�=����콹�_|��J�i��C���c�2)�Vt[E�O����������J׶5�=]�?Dq���׉p��]�FK\e6�iX%Q��l/ {j�k�	9���Αi�j}�V}�����9}�H<Prկ������u������I���]���`hUm�ֲ�x���÷!���a���K�����ak�ᶊ�O����A
5�ہ^7-�Ye���ExӦX�����h�
xbm��O��G�!� ��"����H��I��{��C��3�^�}���b1q��4�_�R����}Z����Wz�D*.N��T,?4p��L?j��&�y��O�2�Sp�2eꡔ򄋋U���][ �gkC���HU���h�s[D0�����\س��2h�a�E��܆��S���Zp�	_.a��n-6f�@mΣN@���K'Xط!��X�)(1�rz��e¢͝-/P��C�c��0f�7t"���ɡ<O������z�����Ho�%�t�MO�X���#�
�n�5�5��
wԇ�-W���K�L\�W����T�/q).4���
��!���o�2�� �J�e��!�)���yP�߆&��H����tR�:��C���F_Z��fG�|�>�tD�TwY����Ʉ�:e���t��Ķ���F �p��_���_l�sl�X��I�U�rK*���@�7���݅\�m��
Llj��X�x�b�	W�ES�)!�u���>�	�@�����hΚ����Dކ��X>�8�$<J���&E�mC���O�@"���h�7��Z7b4b� ������.V6٨#X�9��c��W%�C7�\A��R���������(��24.	��I_��:!�6�e��=tV���hnOk7
��~�A'��N9`&�{$���[��q(#F�j���I*�K�	�6��_��n��&�-��E8��y�&�b���Y�f|:��|bߣ;]Jr!K����`2!4+�a6�� ������F�����CW���p��.��k���ʬ6�A��b�_V��!8�)��]���d�� ۪�1��9'Ov��}Bd��z�&��kFj�e^�B�Է���(���=$��^�Zl�$��.�cԇ~������K!#�9��<�5g�}Kz��&"��$�mˌ�W���9�uJ̹�c˺/��G�\<�\�(��s�d_f2v�N�C=R�rn��{d��;�R .~h�"�)���l{wd���ʀ�;p]��J����n��7�a��@�[}N,ߩ���i}O�J�=�u81}P������	����6F����1¯~$��>!7�U)�����+�e}-�:�P�s�*fȂ����S[��΂�5Ȃe�z8�,��Ů
�ې�=h�:L�$5	�$<�)jf�=�Bj��߂�`M����C���X����SA)b�]/L�0��2j<T/�!��K�ys2��׷��Pd⺁���dꢼ��cnCuR �T"ҩ\&9�EyU�N�X�:a?��}���B�4����2�p8A{�ϕ0fWr�f5��^��O-"��HS���/ڴ;VBx>�A-�LO�;����N
�Vh��͞��ʪ%�G�h�鞺hY*�J��zV�
Q�ٴ�Mf��C��O.&�q~m!��蓶���@��m�I��dM;���6D\ޢ� 5s��˜s���;�a�X`N|2
M]��?:��6t�R���R'��C���;���ư���C3�r���U���EO�H.X�	����!�֖W0ڬt۵6d~��z+����+�i4�:����_}�l��6���K��m��U����p��1_���t��Kc��ޜ�� ������@�������R�i��I%�T��5���(hJ���ԭ��!e�/Bє�Pn�Y���s�/}:։J[��9�Y�ه���`Nh�4��84N��oC�y��js�X�F䆜���=��W:�O�����+q&� �G6�J��y~����.�mq�Ʉٜ`N�n����+��c'fFtk��">��%�Ct췅u0�d ���uC��s��J�TUVv�@q��f�s24�~�S��� wp ?�Iy��U��xR��L��_�y���k!���U���r~�E�cS߇����N�Vz��x���u� ��I�LS��:i���`�c%��s#5D�7��:��q�XC+��lX��L��p'X�J���T��W1'=�L�C0��&n����fE�%�"��.tΆ0�~0���
��3�G�dQ�<b��Q"���x�9�o��޵��{!vB*�|��[�Ӆ��c�߱�r�u(�h��I�v��^�oCM�_��R���h�YT�4��ӶY����7�Stn��v�C ��k�ao�Y��(�˟vV�z#MhXg�X��5����&���).� l�BT>9Ԣ���C�g���֮H��CIq�.`�g�YnZ�'<�mC#�\���}�p�,4�Hޢ�I�>4)`@G�px�������Gۇ*�1<�Q-����i��>�">1 �C�kN�����E��O�4
��!�9���R7�!#�D 3 FB��[�k����> 0NV`�;XD��83c]�,2��i�@+b�U�#��57F�-Na�<�!�&̭��+����x�L�bc�(.:׎hsM���m;��m�VN�&3D��	:�=������>bJ�!49*3�7�֣�����h�C�B? G�i��k��~�ۘ%��F�$��;p�=Z�kK;���B;X*8#�ۃI�����۶b;A����	+R���TH���>�R�
��`BZ���02��U��JLʹT�u�5�au�m�FF�-^l��Lt~��l���P���j-�q��L����Z4ufS�G�)�fWE�\�+F:`�=���dg]t
ϩ!��%_�bDc�c�
n�.q�l¼�"��d�>U0�@\H��c��g/ew�֣�'c��E7���q��"��F��x�Fv�G�sl��ls�m�6)j!���%���S{+箑��ل~��wV��^����§8{�`i��A�lֽ�o("Z��hJ������9[���E6"pB��2��c>����Qp�ː��$���R�O�sKr��7�6D`01�}d����͞�m�Z����B+�́��aIl��m�A�"��m�Ee�hNm��)�'��ec�� �M�����k_�s�3�m�Q�bD�{����\���\�!�qԧC���}�*	��i���֖�A��l#�9���ΰ�ױKR��Eb��dp���5}̷�o�7ėE'-��G���+��k��M��n�H�X�'��ŧ�CF�e �0���"C��t� �AQe]��L:�Г�T���8���3��ed��]�j��3uF�L}�.L*й��\�����h�b����i�j���u��P�&Dg�k�ٔ����,�W����&lD�:+���sζ�ytm�%��!�=7��V�m�0��ʓ�l�:z�R����J�>�J��X��#�������`�����nXU��f�s6��m�$�.�D����.t�6-v���2z���R{���u�M��-�㠈��;Cg�B�n{���=���F�Ir��L��B]	a����K�ί�0($6��`��Egx$�� =��� z��ٗkv��5Uv_Yjl/^��[9eE�`O��dH� <zih�!;��$��u"��Jl���k:�nwct �  �j���ϖ���t��վY��a�v�M�ܧ+]� [�'�f4)��7�5GJ����#��X�I2K:�e� �݆j��s��MV-���鮥����qC�'I�L�D���n�������ê*���_����*K�����2�<�	�s�_�b��ܓ?�]��g|=q�ȅ�D2���թ%�v5>Mڇ�����ik��hT�c���9w��weËW>Z���<�kG�!m~�,At{D�.O�\;2�>���h��Kc�0r�J(���؇"e7U�J�ϵ�0i�1��',pg��/I��Wf�Q랾��Eg^֘�c�R���������ڈ�/��Ʌ<ҩ�B�l�c�^F��/d{�lkR��={�6�l:_Q�,�?U𨎇�]� ��������>X���ږ��h,�����q�3���k6���O=��m��-��;*_O/����3��Ƚ�z{Bo �$���[0��2=�y���~#ln>7̹�y���8�"\N�Uֶ��[5C�e���e�
�LfauO-��AP�#������w!��P�_���#���j���Z�}�(։�����s�umW��x4�D6O#�������n�=!H/��K(3���w!�9�8���T��Y0i$��S+���fdA�Y� f���l�k�����5�ʧY�G��9�y�j&3��| 0�#��]���^~�e]��D+k���K�|UҲ@�F\��PY��"�=+*=�)1O�!�7��fe�h�,��iS��W�8_hͶh0؆Dh&v"������ـ�CQ���S��Bk6b���q,�[/���q��9�g����˳{�ߨl�f�5�������д	�̓�R�Ӆ�٠h~`��|�Zg"����*v��P���1�aw =���F���B��	�߄=��f��U�a��3��)������s�ޫ�;�@��?�0�ր���S0�3O���6�l�1���v��@z�\�,��|,�'�iQ?� ��a�!%+�����Ң�Yچ���yP�����%C�lQ�A�G��S*ǃAvs�tM��֓a���7��ؕ�|��-�
�,-^��^�u���.�{(�_�1�t���G5?�2���8�t�}�ņZGߣ!�T���n,z�҃����~c|}��n���]��w|�臘��:��D�U/����;x��J~T�r�d'���>Hq_�1'�dn��]w|�b|i@�Lǻ�N;r�N�G��ZA���c�
%��|t�k|�ran4�}���]����){�V�W����<I�/�nm.����e�#�w�r�
8�.τ=,����x���v�|��s��x��p��R��%��b|������Z�a�+W�k�����G�_�m=h3%Ϣ\��mO\֚ \VΩ��9E�BV(�^��=�U����q�-��q0�x�G�7�j��;��l�A_�/RA�""o�:�0}oq)QF�˔�U�7���|`��/��
�p�0B=�!�P4����K#BTUb��P���_�hŷ��b��{��xu���$�	�����:@Z�,=�I>Sn;b���:�h��HrPu��E)�[2|�_���g�yk�9^|��t߆4���v|��n(:�m.{�,���4Y�ӹ��b(?V�'�d�0D�p�Y�\�.����A#Z "L��Ў��-��Z�1H�`,HYݍ)�O����� ��M��et��OF�)~b�(V?����W����,�},��pwxX���n5�J�XT?����1�#=ގ߄/���?��ٰǘ%f ؑ[��}�Ķ���5z�J��p~igc6�ǥ�:���B+��/�f����B����x&|�]9�Y,þ��dӳ~y��{B�]ڵ/"��AR��V���Y{��@^m������=$z��?�ޖ?ۂ�L��w �i:~$��k�պ���KiiO���;�@�HA24�gF����B��F΢k[`ж��q���	��A�#7�/�VF<���ot�4��HE�е��ֱ���kAT��fĝ�=g]o�F;{[?j��PÝ3Z��!�m`��٪�<FN����Ok����QUN�k6�/���U^g���P~��ξDe��g��l�A�̖=~�и��[�v)b;E->A �ڢ��]��9�={�<�O�(�+�Y��֙��5����}m&� �f�.�r��#2�X��ͳ�v5lAg��kۆm�>dQO�x��8{̻�9ۆn��#�����;����e�E}3kњ=�>Y�}̻�ʹ��hG���	�d�����;��}�ҫ�F� J�HmޕV��j)[�Ext��Hh�|����Y��>�[h,=�+����ז����KRU{�8]�܈\�m��_��wVa�����̍�5��䃢O9 _�)%�?_�s���#�S3LE���un��"~���0�~nD�٣|�(J�l�U�<�5�G!���K( �s� ŀv�Շ�Y�s߮c+/�w�� l�i~0�-y����6�2�0c/X#2�~g�ֶ���(���/
o��G���P4k������:K%|l�W8�����'s"l~eO�Ќ�/_hͶ�E!�VVأ�?$y&��:	�C�_�a���Ю������e�^u=�	��z��s���Bm/q��ϋ�t�7��>�k:�{���l�f�s��6����؇f�_������h����x�6�WBL�[U�6X�d.+ow��w��],k|n�4��W����S�ao���'��![��i%�ym,ZLvqZ�S��+�T�'�/�B�C�ίlNV���]����P���2�-�/G!f�J�B��͝�!�GŔ����ϵ�ZO��>�5s�GuR.��������D>���k��E!��wT��5�=���ru�E`%6H޵5�#T��œN���	w���7�]݇�K:=��y���r      -      x�3�44�4�3�3������ �K     