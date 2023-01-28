PGDMP                         x            optuna    12.4    12.2 U    c           0    0    ENCODING    ENCODING        SET client_encoding = 'UTF8';
                      false            d           0    0 
   STDSTRINGS 
   STDSTRINGS     (   SET standard_conforming_strings = 'on';
                      false            e           0    0 
   SEARCHPATH 
   SEARCHPATH     8   SELECT pg_catalog.set_config('search_path', '', false);
                      false            f           1262    16409    optuna    DATABASE     x   CREATE DATABASE optuna WITH TEMPLATE = template0 ENCODING = 'UTF8' LC_COLLATE = 'en_US.UTF-8' LC_CTYPE = 'en_US.UTF-8';
    DROP DATABASE optuna;
                hannan    false                        2615    2200    public    SCHEMA        CREATE SCHEMA public;
    DROP SCHEMA public;
                hannan    false            g           0    0    SCHEMA public    COMMENT     6   COMMENT ON SCHEMA public IS 'standard public schema';
                   hannan    false    3            h           0    0    SCHEMA public    ACL     �   REVOKE ALL ON SCHEMA public FROM rdsadmin;
REVOKE ALL ON SCHEMA public FROM PUBLIC;
GRANT ALL ON SCHEMA public TO hannan;
GRANT ALL ON SCHEMA public TO PUBLIC;
                   hannan    false    3            )           1247    16411    studydirection    TYPE     ]   CREATE TYPE public.studydirection AS ENUM (
    'NOT_SET',
    'MINIMIZE',
    'MAXIMIZE'
);
 !   DROP TYPE public.studydirection;
       public          hannan    false    3            ,           1247    16418 
   trialstate    TYPE     r   CREATE TYPE public.trialstate AS ENUM (
    'RUNNING',
    'COMPLETE',
    'PRUNED',
    'FAIL',
    'WAITING'
);
    DROP TYPE public.trialstate;
       public          hannan    false    3            �            1259    16565    alembic_version    TABLE     X   CREATE TABLE public.alembic_version (
    version_num character varying(32) NOT NULL
);
 #   DROP TABLE public.alembic_version;
       public         heap    hannan    false    3            �            1259    16431    studies    TABLE     �   CREATE TABLE public.studies (
    study_id integer NOT NULL,
    study_name character varying(512) NOT NULL,
    direction public.studydirection NOT NULL
);
    DROP TABLE public.studies;
       public         heap    hannan    false    3    553            �            1259    16429    studies_study_id_seq    SEQUENCE     �   CREATE SEQUENCE public.studies_study_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 +   DROP SEQUENCE public.studies_study_id_seq;
       public          hannan    false    203    3            i           0    0    studies_study_id_seq    SEQUENCE OWNED BY     M   ALTER SEQUENCE public.studies_study_id_seq OWNED BY public.studies.study_id;
          public          hannan    false    202            �            1259    16467    study_system_attributes    TABLE     �   CREATE TABLE public.study_system_attributes (
    study_system_attribute_id integer NOT NULL,
    study_id integer,
    key character varying(512),
    value_json character varying(2048)
);
 +   DROP TABLE public.study_system_attributes;
       public         heap    hannan    false    3            �            1259    16465 5   study_system_attributes_study_system_attribute_id_seq    SEQUENCE     �   CREATE SEQUENCE public.study_system_attributes_study_system_attribute_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 L   DROP SEQUENCE public.study_system_attributes_study_system_attribute_id_seq;
       public          hannan    false    3    208            j           0    0 5   study_system_attributes_study_system_attribute_id_seq    SEQUENCE OWNED BY     �   ALTER SEQUENCE public.study_system_attributes_study_system_attribute_id_seq OWNED BY public.study_system_attributes.study_system_attribute_id;
          public          hannan    false    207            �            1259    16449    study_user_attributes    TABLE     �   CREATE TABLE public.study_user_attributes (
    study_user_attribute_id integer NOT NULL,
    study_id integer,
    key character varying(512),
    value_json character varying(2048)
);
 )   DROP TABLE public.study_user_attributes;
       public         heap    hannan    false    3            �            1259    16447 1   study_user_attributes_study_user_attribute_id_seq    SEQUENCE     �   CREATE SEQUENCE public.study_user_attributes_study_user_attribute_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 H   DROP SEQUENCE public.study_user_attributes_study_user_attribute_id_seq;
       public          hannan    false    3    206            k           0    0 1   study_user_attributes_study_user_attribute_id_seq    SEQUENCE OWNED BY     �   ALTER SEQUENCE public.study_user_attributes_study_user_attribute_id_seq OWNED BY public.study_user_attributes.study_user_attribute_id;
          public          hannan    false    205            �            1259    16534    trial_params    TABLE     �   CREATE TABLE public.trial_params (
    param_id integer NOT NULL,
    trial_id integer,
    param_name character varying(512),
    param_value double precision,
    distribution_json character varying(2048)
);
     DROP TABLE public.trial_params;
       public         heap    hannan    false    3            �            1259    16532    trial_params_param_id_seq    SEQUENCE     �   CREATE SEQUENCE public.trial_params_param_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 0   DROP SEQUENCE public.trial_params_param_id_seq;
       public          hannan    false    3    216            l           0    0    trial_params_param_id_seq    SEQUENCE OWNED BY     W   ALTER SEQUENCE public.trial_params_param_id_seq OWNED BY public.trial_params.param_id;
          public          hannan    false    215            �            1259    16516    trial_system_attributes    TABLE     �   CREATE TABLE public.trial_system_attributes (
    trial_system_attribute_id integer NOT NULL,
    trial_id integer,
    key character varying(512),
    value_json character varying(2048)
);
 +   DROP TABLE public.trial_system_attributes;
       public         heap    hannan    false    3            �            1259    16514 5   trial_system_attributes_trial_system_attribute_id_seq    SEQUENCE     �   CREATE SEQUENCE public.trial_system_attributes_trial_system_attribute_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 L   DROP SEQUENCE public.trial_system_attributes_trial_system_attribute_id_seq;
       public          hannan    false    214    3            m           0    0 5   trial_system_attributes_trial_system_attribute_id_seq    SEQUENCE OWNED BY     �   ALTER SEQUENCE public.trial_system_attributes_trial_system_attribute_id_seq OWNED BY public.trial_system_attributes.trial_system_attribute_id;
          public          hannan    false    213            �            1259    16498    trial_user_attributes    TABLE     �   CREATE TABLE public.trial_user_attributes (
    trial_user_attribute_id integer NOT NULL,
    trial_id integer,
    key character varying(512),
    value_json character varying(2048)
);
 )   DROP TABLE public.trial_user_attributes;
       public         heap    hannan    false    3            �            1259    16496 1   trial_user_attributes_trial_user_attribute_id_seq    SEQUENCE     �   CREATE SEQUENCE public.trial_user_attributes_trial_user_attribute_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 H   DROP SEQUENCE public.trial_user_attributes_trial_user_attribute_id_seq;
       public          hannan    false    3    212            n           0    0 1   trial_user_attributes_trial_user_attribute_id_seq    SEQUENCE OWNED BY     �   ALTER SEQUENCE public.trial_user_attributes_trial_user_attribute_id_seq OWNED BY public.trial_user_attributes.trial_user_attribute_id;
          public          hannan    false    211            �            1259    16552    trial_values    TABLE     �   CREATE TABLE public.trial_values (
    trial_value_id integer NOT NULL,
    trial_id integer,
    step integer,
    value double precision
);
     DROP TABLE public.trial_values;
       public         heap    hannan    false    3            �            1259    16550    trial_values_trial_value_id_seq    SEQUENCE     �   CREATE SEQUENCE public.trial_values_trial_value_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 6   DROP SEQUENCE public.trial_values_trial_value_id_seq;
       public          hannan    false    3    218            o           0    0    trial_values_trial_value_id_seq    SEQUENCE OWNED BY     c   ALTER SEQUENCE public.trial_values_trial_value_id_seq OWNED BY public.trial_values.trial_value_id;
          public          hannan    false    217            �            1259    16485    trials    TABLE       CREATE TABLE public.trials (
    trial_id integer NOT NULL,
    number integer,
    study_id integer,
    state public.trialstate NOT NULL,
    value double precision,
    datetime_start timestamp without time zone,
    datetime_complete timestamp without time zone
);
    DROP TABLE public.trials;
       public         heap    hannan    false    3    556            �            1259    16483    trials_trial_id_seq    SEQUENCE     �   CREATE SEQUENCE public.trials_trial_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 *   DROP SEQUENCE public.trials_trial_id_seq;
       public          hannan    false    3    210            p           0    0    trials_trial_id_seq    SEQUENCE OWNED BY     K   ALTER SEQUENCE public.trials_trial_id_seq OWNED BY public.trials.trial_id;
          public          hannan    false    209            �            1259    16441    version_info    TABLE     �   CREATE TABLE public.version_info (
    version_info_id integer NOT NULL,
    schema_version integer,
    library_version character varying(256),
    CONSTRAINT version_info_version_info_id_check CHECK ((version_info_id = 1))
);
     DROP TABLE public.version_info;
       public         heap    hannan    false    3            �           2604    16434    studies study_id    DEFAULT     t   ALTER TABLE ONLY public.studies ALTER COLUMN study_id SET DEFAULT nextval('public.studies_study_id_seq'::regclass);
 ?   ALTER TABLE public.studies ALTER COLUMN study_id DROP DEFAULT;
       public          hannan    false    203    202    203            �           2604    16470 1   study_system_attributes study_system_attribute_id    DEFAULT     �   ALTER TABLE ONLY public.study_system_attributes ALTER COLUMN study_system_attribute_id SET DEFAULT nextval('public.study_system_attributes_study_system_attribute_id_seq'::regclass);
 `   ALTER TABLE public.study_system_attributes ALTER COLUMN study_system_attribute_id DROP DEFAULT;
       public          hannan    false    207    208    208            �           2604    16452 -   study_user_attributes study_user_attribute_id    DEFAULT     �   ALTER TABLE ONLY public.study_user_attributes ALTER COLUMN study_user_attribute_id SET DEFAULT nextval('public.study_user_attributes_study_user_attribute_id_seq'::regclass);
 \   ALTER TABLE public.study_user_attributes ALTER COLUMN study_user_attribute_id DROP DEFAULT;
       public          hannan    false    206    205    206            �           2604    16537    trial_params param_id    DEFAULT     ~   ALTER TABLE ONLY public.trial_params ALTER COLUMN param_id SET DEFAULT nextval('public.trial_params_param_id_seq'::regclass);
 D   ALTER TABLE public.trial_params ALTER COLUMN param_id DROP DEFAULT;
       public          hannan    false    216    215    216            �           2604    16519 1   trial_system_attributes trial_system_attribute_id    DEFAULT     �   ALTER TABLE ONLY public.trial_system_attributes ALTER COLUMN trial_system_attribute_id SET DEFAULT nextval('public.trial_system_attributes_trial_system_attribute_id_seq'::regclass);
 `   ALTER TABLE public.trial_system_attributes ALTER COLUMN trial_system_attribute_id DROP DEFAULT;
       public          hannan    false    214    213    214            �           2604    16501 -   trial_user_attributes trial_user_attribute_id    DEFAULT     �   ALTER TABLE ONLY public.trial_user_attributes ALTER COLUMN trial_user_attribute_id SET DEFAULT nextval('public.trial_user_attributes_trial_user_attribute_id_seq'::regclass);
 \   ALTER TABLE public.trial_user_attributes ALTER COLUMN trial_user_attribute_id DROP DEFAULT;
       public          hannan    false    211    212    212            �           2604    16555    trial_values trial_value_id    DEFAULT     �   ALTER TABLE ONLY public.trial_values ALTER COLUMN trial_value_id SET DEFAULT nextval('public.trial_values_trial_value_id_seq'::regclass);
 J   ALTER TABLE public.trial_values ALTER COLUMN trial_value_id DROP DEFAULT;
       public          hannan    false    218    217    218            �           2604    16488    trials trial_id    DEFAULT     r   ALTER TABLE ONLY public.trials ALTER COLUMN trial_id SET DEFAULT nextval('public.trials_trial_id_seq'::regclass);
 >   ALTER TABLE public.trials ALTER COLUMN trial_id DROP DEFAULT;
       public          hannan    false    209    210    210            `          0    16565    alembic_version 
   TABLE DATA           6   COPY public.alembic_version (version_num) FROM stdin;
    public          hannan    false    219            P          0    16431    studies 
   TABLE DATA           B   COPY public.studies (study_id, study_name, direction) FROM stdin;
    public          hannan    false    203            U          0    16467    study_system_attributes 
   TABLE DATA           g   COPY public.study_system_attributes (study_system_attribute_id, study_id, key, value_json) FROM stdin;
    public          hannan    false    208            S          0    16449    study_user_attributes 
   TABLE DATA           c   COPY public.study_user_attributes (study_user_attribute_id, study_id, key, value_json) FROM stdin;
    public          hannan    false    206            ]          0    16534    trial_params 
   TABLE DATA           f   COPY public.trial_params (param_id, trial_id, param_name, param_value, distribution_json) FROM stdin;
    public          hannan    false    216            [          0    16516    trial_system_attributes 
   TABLE DATA           g   COPY public.trial_system_attributes (trial_system_attribute_id, trial_id, key, value_json) FROM stdin;
    public          hannan    false    214            Y          0    16498    trial_user_attributes 
   TABLE DATA           c   COPY public.trial_user_attributes (trial_user_attribute_id, trial_id, key, value_json) FROM stdin;
    public          hannan    false    212            _          0    16552    trial_values 
   TABLE DATA           M   COPY public.trial_values (trial_value_id, trial_id, step, value) FROM stdin;
    public          hannan    false    218            W          0    16485    trials 
   TABLE DATA           m   COPY public.trials (trial_id, number, study_id, state, value, datetime_start, datetime_complete) FROM stdin;
    public          hannan    false    210            Q          0    16441    version_info 
   TABLE DATA           X   COPY public.version_info (version_info_id, schema_version, library_version) FROM stdin;
    public          hannan    false    204            q           0    0    studies_study_id_seq    SEQUENCE SET     C   SELECT pg_catalog.setval('public.studies_study_id_seq', 24, true);
          public          hannan    false    202            r           0    0 5   study_system_attributes_study_system_attribute_id_seq    SEQUENCE SET     d   SELECT pg_catalog.setval('public.study_system_attributes_study_system_attribute_id_seq', 1, false);
          public          hannan    false    207            s           0    0 1   study_user_attributes_study_user_attribute_id_seq    SEQUENCE SET     `   SELECT pg_catalog.setval('public.study_user_attributes_study_user_attribute_id_seq', 1, false);
          public          hannan    false    205            t           0    0    trial_params_param_id_seq    SEQUENCE SET     J   SELECT pg_catalog.setval('public.trial_params_param_id_seq', 1481, true);
          public          hannan    false    215            u           0    0 5   trial_system_attributes_trial_system_attribute_id_seq    SEQUENCE SET     d   SELECT pg_catalog.setval('public.trial_system_attributes_trial_system_attribute_id_seq', 25, true);
          public          hannan    false    213            v           0    0 1   trial_user_attributes_trial_user_attribute_id_seq    SEQUENCE SET     `   SELECT pg_catalog.setval('public.trial_user_attributes_trial_user_attribute_id_seq', 1, false);
          public          hannan    false    211            w           0    0    trial_values_trial_value_id_seq    SEQUENCE SET     O   SELECT pg_catalog.setval('public.trial_values_trial_value_id_seq', 100, true);
          public          hannan    false    217            x           0    0    trials_trial_id_seq    SEQUENCE SET     C   SELECT pg_catalog.setval('public.trials_trial_id_seq', 193, true);
          public          hannan    false    209            �           2606    16569 #   alembic_version alembic_version_pkc 
   CONSTRAINT     j   ALTER TABLE ONLY public.alembic_version
    ADD CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num);
 M   ALTER TABLE ONLY public.alembic_version DROP CONSTRAINT alembic_version_pkc;
       public            hannan    false    219            �           2606    16439    studies studies_pkey 
   CONSTRAINT     X   ALTER TABLE ONLY public.studies
    ADD CONSTRAINT studies_pkey PRIMARY KEY (study_id);
 >   ALTER TABLE ONLY public.studies DROP CONSTRAINT studies_pkey;
       public            hannan    false    203            �           2606    16475 4   study_system_attributes study_system_attributes_pkey 
   CONSTRAINT     �   ALTER TABLE ONLY public.study_system_attributes
    ADD CONSTRAINT study_system_attributes_pkey PRIMARY KEY (study_system_attribute_id);
 ^   ALTER TABLE ONLY public.study_system_attributes DROP CONSTRAINT study_system_attributes_pkey;
       public            hannan    false    208            �           2606    16477 @   study_system_attributes study_system_attributes_study_id_key_key 
   CONSTRAINT     �   ALTER TABLE ONLY public.study_system_attributes
    ADD CONSTRAINT study_system_attributes_study_id_key_key UNIQUE (study_id, key);
 j   ALTER TABLE ONLY public.study_system_attributes DROP CONSTRAINT study_system_attributes_study_id_key_key;
       public            hannan    false    208    208            �           2606    16457 0   study_user_attributes study_user_attributes_pkey 
   CONSTRAINT     �   ALTER TABLE ONLY public.study_user_attributes
    ADD CONSTRAINT study_user_attributes_pkey PRIMARY KEY (study_user_attribute_id);
 Z   ALTER TABLE ONLY public.study_user_attributes DROP CONSTRAINT study_user_attributes_pkey;
       public            hannan    false    206            �           2606    16459 <   study_user_attributes study_user_attributes_study_id_key_key 
   CONSTRAINT     �   ALTER TABLE ONLY public.study_user_attributes
    ADD CONSTRAINT study_user_attributes_study_id_key_key UNIQUE (study_id, key);
 f   ALTER TABLE ONLY public.study_user_attributes DROP CONSTRAINT study_user_attributes_study_id_key_key;
       public            hannan    false    206    206            �           2606    16542    trial_params trial_params_pkey 
   CONSTRAINT     b   ALTER TABLE ONLY public.trial_params
    ADD CONSTRAINT trial_params_pkey PRIMARY KEY (param_id);
 H   ALTER TABLE ONLY public.trial_params DROP CONSTRAINT trial_params_pkey;
       public            hannan    false    216            �           2606    16544 1   trial_params trial_params_trial_id_param_name_key 
   CONSTRAINT     |   ALTER TABLE ONLY public.trial_params
    ADD CONSTRAINT trial_params_trial_id_param_name_key UNIQUE (trial_id, param_name);
 [   ALTER TABLE ONLY public.trial_params DROP CONSTRAINT trial_params_trial_id_param_name_key;
       public            hannan    false    216    216            �           2606    16524 4   trial_system_attributes trial_system_attributes_pkey 
   CONSTRAINT     �   ALTER TABLE ONLY public.trial_system_attributes
    ADD CONSTRAINT trial_system_attributes_pkey PRIMARY KEY (trial_system_attribute_id);
 ^   ALTER TABLE ONLY public.trial_system_attributes DROP CONSTRAINT trial_system_attributes_pkey;
       public            hannan    false    214            �           2606    16526 @   trial_system_attributes trial_system_attributes_trial_id_key_key 
   CONSTRAINT     �   ALTER TABLE ONLY public.trial_system_attributes
    ADD CONSTRAINT trial_system_attributes_trial_id_key_key UNIQUE (trial_id, key);
 j   ALTER TABLE ONLY public.trial_system_attributes DROP CONSTRAINT trial_system_attributes_trial_id_key_key;
       public            hannan    false    214    214            �           2606    16506 0   trial_user_attributes trial_user_attributes_pkey 
   CONSTRAINT     �   ALTER TABLE ONLY public.trial_user_attributes
    ADD CONSTRAINT trial_user_attributes_pkey PRIMARY KEY (trial_user_attribute_id);
 Z   ALTER TABLE ONLY public.trial_user_attributes DROP CONSTRAINT trial_user_attributes_pkey;
       public            hannan    false    212            �           2606    16508 <   trial_user_attributes trial_user_attributes_trial_id_key_key 
   CONSTRAINT     �   ALTER TABLE ONLY public.trial_user_attributes
    ADD CONSTRAINT trial_user_attributes_trial_id_key_key UNIQUE (trial_id, key);
 f   ALTER TABLE ONLY public.trial_user_attributes DROP CONSTRAINT trial_user_attributes_trial_id_key_key;
       public            hannan    false    212    212            �           2606    16557    trial_values trial_values_pkey 
   CONSTRAINT     h   ALTER TABLE ONLY public.trial_values
    ADD CONSTRAINT trial_values_pkey PRIMARY KEY (trial_value_id);
 H   ALTER TABLE ONLY public.trial_values DROP CONSTRAINT trial_values_pkey;
       public            hannan    false    218            �           2606    16559 +   trial_values trial_values_trial_id_step_key 
   CONSTRAINT     p   ALTER TABLE ONLY public.trial_values
    ADD CONSTRAINT trial_values_trial_id_step_key UNIQUE (trial_id, step);
 U   ALTER TABLE ONLY public.trial_values DROP CONSTRAINT trial_values_trial_id_step_key;
       public            hannan    false    218    218            �           2606    16490    trials trials_pkey 
   CONSTRAINT     V   ALTER TABLE ONLY public.trials
    ADD CONSTRAINT trials_pkey PRIMARY KEY (trial_id);
 <   ALTER TABLE ONLY public.trials DROP CONSTRAINT trials_pkey;
       public            hannan    false    210            �           2606    16446    version_info version_info_pkey 
   CONSTRAINT     i   ALTER TABLE ONLY public.version_info
    ADD CONSTRAINT version_info_pkey PRIMARY KEY (version_info_id);
 H   ALTER TABLE ONLY public.version_info DROP CONSTRAINT version_info_pkey;
       public            hannan    false    204            �           1259    16440    ix_studies_study_name    INDEX     V   CREATE UNIQUE INDEX ix_studies_study_name ON public.studies USING btree (study_name);
 )   DROP INDEX public.ix_studies_study_name;
       public            hannan    false    203            �           2606    16478 =   study_system_attributes study_system_attributes_study_id_fkey    FK CONSTRAINT     �   ALTER TABLE ONLY public.study_system_attributes
    ADD CONSTRAINT study_system_attributes_study_id_fkey FOREIGN KEY (study_id) REFERENCES public.studies(study_id);
 g   ALTER TABLE ONLY public.study_system_attributes DROP CONSTRAINT study_system_attributes_study_id_fkey;
       public          hannan    false    208    203    3755            �           2606    16460 9   study_user_attributes study_user_attributes_study_id_fkey    FK CONSTRAINT     �   ALTER TABLE ONLY public.study_user_attributes
    ADD CONSTRAINT study_user_attributes_study_id_fkey FOREIGN KEY (study_id) REFERENCES public.studies(study_id);
 c   ALTER TABLE ONLY public.study_user_attributes DROP CONSTRAINT study_user_attributes_study_id_fkey;
       public          hannan    false    203    3755    206            �           2606    16545 '   trial_params trial_params_trial_id_fkey    FK CONSTRAINT     �   ALTER TABLE ONLY public.trial_params
    ADD CONSTRAINT trial_params_trial_id_fkey FOREIGN KEY (trial_id) REFERENCES public.trials(trial_id);
 Q   ALTER TABLE ONLY public.trial_params DROP CONSTRAINT trial_params_trial_id_fkey;
       public          hannan    false    210    3767    216            �           2606    16527 =   trial_system_attributes trial_system_attributes_trial_id_fkey    FK CONSTRAINT     �   ALTER TABLE ONLY public.trial_system_attributes
    ADD CONSTRAINT trial_system_attributes_trial_id_fkey FOREIGN KEY (trial_id) REFERENCES public.trials(trial_id);
 g   ALTER TABLE ONLY public.trial_system_attributes DROP CONSTRAINT trial_system_attributes_trial_id_fkey;
       public          hannan    false    214    210    3767            �           2606    16509 9   trial_user_attributes trial_user_attributes_trial_id_fkey    FK CONSTRAINT     �   ALTER TABLE ONLY public.trial_user_attributes
    ADD CONSTRAINT trial_user_attributes_trial_id_fkey FOREIGN KEY (trial_id) REFERENCES public.trials(trial_id);
 c   ALTER TABLE ONLY public.trial_user_attributes DROP CONSTRAINT trial_user_attributes_trial_id_fkey;
       public          hannan    false    3767    212    210            �           2606    16560 '   trial_values trial_values_trial_id_fkey    FK CONSTRAINT     �   ALTER TABLE ONLY public.trial_values
    ADD CONSTRAINT trial_values_trial_id_fkey FOREIGN KEY (trial_id) REFERENCES public.trials(trial_id);
 Q   ALTER TABLE ONLY public.trial_values DROP CONSTRAINT trial_values_trial_id_fkey;
       public          hannan    false    3767    210    218            �           2606    16491    trials trials_study_id_fkey    FK CONSTRAINT     �   ALTER TABLE ONLY public.trials
    ADD CONSTRAINT trials_study_id_fkey FOREIGN KEY (study_id) REFERENCES public.studies(study_id);
 E   ALTER TABLE ONLY public.trials DROP CONSTRAINT trials_study_id_fkey;
       public          hannan    false    210    3755    203            `      x�+3�3�3�K����� ��      P   u   x�m�1!@���� x cB&�k�&r{e-h��Ϸ��)�
x!&G�*�K��z�n0:rD� �0���<x6��  ��ZcOm�0>��[z���0>h�`�����Z�/%7s      U      x������ � �      S      x������ � �      ]      x�͝]�\Ǒ��G�B�kQ���w��!_��` ԈD� d�zc���33� @}&K�"ŧ�P�t֩7�����������7�o�����������������������~������<����W����������^~8~{�������7��6��~z��O���<��?~����݇�o�����/����F����� R&�Ǘ?���7��^~������?�|��?�{�����/���?��1׏ǟ<������$�$�y7^dZ�;g��]������oy�v�_�_���ؤ�z�ᇟ��������M��������=�|�l�O������<4��������2�L�/o���� *0��/߿}���9s^����	�����@KL��/_���������o�1�젂W����s�c�Ǉ���р�1)o�?|���?��7�ݿ����7o�/=����?�����՛����/����z�d�����?�r�7K���O
�e ��N{�F2Q͂#���p��pd=g/F��<K�:�r�t�����Wk��+�ݎ���(1���͟�ϊ�
��b���%�� p�N�_uL��Su�O.hS˝t�r:Qͪc��=��DO��.�,qQApչ�$]�����W����qa<�T'������ٗT'`��'�Iuү:�'��Ntm���&�{��y����7�Nb���uN�1��Ceh�p�SM'9A[�:)��Lw�����Ӂ����n��~�tJ�����/�N�i����M�2Y;�:�'��N�ΚL�>QͦӘ�=�W͉>L�5_����V�*:�ɹ�͟����{Dg��l{Lg��z�=��L ���d=��f���̞L�&:�Y���b��Eg9���n���(:��~(��L�R�˒�SM�c��Lws����r�����V�Ͳ�[M�
`���m�v.^ݟT�f:��h2��D5��k2�,^cL��x�f.<�s=�tAte�۷?��t� ��t!ζ�.���1]�=5]�d���"�T�f�Ȼl2]�D5�.�d�;a}��&z�.^&/�&��H$OҖ�t) �u��OTw��ϦL��q�5�����|Iu������߮��'��.뮚TWc��UW4���nߧ�6�O�Qԑq)���L����%���+�'NW��l[��c�6�HT 8Mw�<���*'k�DՓ
�u�?O�+S7��T�`P��Mh��x�B�L��$�8�ӫ;
ؖ�v�p]��,_��V�ш5u/�Y_r�\Ȯ<q����Ѹ�π2wuM1`�$�0�	��̀��?<���3+I�bK�sH֕ �O���$@�5�m�Q��{�ͬ/
�j!{��Оx�_��!@���*����*X � +�{�1��G���
�R�Y�*@v��x����c��ȹF�mC����m�� �$@!�vnk���}�餮f�[�b�n�8�K�S��)3%�J�C&	��<�#��nҟ��>��X���)-d���A۪?������H]�X{c���I�?���*�F��'v�
P�-;����M��F�ol�fn� ��I�&�m=S��b�>�tRW_��-@P7	���(���!��wg��|� ���S�X�&:�Q�� ����\ �,d� ]A����b�>�tRW�y �-@OP7	���
��R���*���M����	Z�&�a�+C���4`�B���(��k����SJ]}	X��@�d��?=��˹>�~I �� I^����!��_ꚶ����H� �t�v��)㢆>��RWoe�m��n�����F�A�+�O����z��Q*Y��tj_�lM��L�B6�+@�Z�U^��翚J�j��{���F��6�Y�H`?�}�����_�X���c��k��,��uuُ��������1��z�2#s]]�cĭ�6���N�1]�#��~88z���k߹��1�54�$?U8@zl�N�n��`�"�i���:�8<z��gH���/�����qqC��p�tۭ8J����Y@ݤ?V���B�͗#)��+�l���b�<�c�5�mO��s��N�q-dO�� m���.fh<e~����c~�떟(�{Nva1�&-75�i�R�{2)��%����T�I�A�[�J������:�i���V�)_���?�J�gC����{�{au�w>$�ب4����64��S�i.v���֠���ƚ�����B6�����3�ȡ�6�����f���4�nZ�Z �t^3���
I�����jK�[-v��|�A��Ӛ���v^�&������9��ϧ��Z?�ċ�[?��=�?��)���&a1,b��+`/�^m�p-�b70hk����mL>s�B60���v�C�cJ����# �6`$��*�(���b����'o�� k�8i����F�O�)k�v.�S�����@�~�����I]��	X� �@�T� �A��5_�,�Gi��)j�H�b7��dj��J����)[Ȧ����g��ȡ�5�����U�u��;	��n��A���zy��pO>W2�-O e�b��O��1mK�Ȱ5o{?d�B��OF���	��������1����!�Z��~2׵��M���K�]�MuX�R�$�k�����u[�� m���!\W�Օ��u[�� m][�?��ءO�<����X� �A��� ��ΜLN�y��XAz���W��'-�kL�ze�ˤ��Cf-$m����m�7��O����?D�n����I��?��ʲJwg}*b ����b7�Obi��$״����B6�Oh[��tQC��t
���CT 붟*�����_��}�pE�\����~��d?�5�}��ZӶ�~6��~F�m���E}��)���1��~f�n��9���F����nJ��"Ӧ"��g��M��ZC�g?k�v��i!����V��\��g?�B����9w���!�n����ϒO��7;�ɷ�'@{���M������~Ak�v�/x!���V��^��g��B����p����&�E��'�<Jɒ���S����X�&�%�1��_򚷝�KY�&����Ui7��/����>$�n�e��IY����R(hd������-v���א��o�����1�)��~e�m�_�E}��)������~U�η'�h?r6.Zk�sO=�A mџ^���5�m�ӡk�6�O�-d��t8h;��#.nhӟ�i���X������G������'����}��"���P��n���>���y۩?�l�h[�GyqC���L�Z>����Eκv�q�z�M�ͩ�ɡ!�f���ڤ?���M�W����C����������CȺ��|�U��͟��C\7��OƝv5|�`��u��D _��꛾X��x��{ѩS���b�ݤ?�5�}���2������5|������qqC��t���C����\eꮆU�h�u�Z��G��O�=�S_�&�i�A��暸��_����;�I������կ�E}������5����n��G�J�|.}9Ƙ�����=��X�&�Y�!���՚��՟��l���@�Z�9_��g?�B�j��Y�h{ˇ����~���Lt_RS|�ίH{���M��Zcڧ�k�v�/h!���V��\�Ч��F����0����&�E ~i��)N�|�m	��E-v��r�!��_Қ���K^�&�����~�5��/���z>4�n�e�z��m��|�/��q�A8�wr�%�=����W������ۯhi���״��_�B6ٯ���+����~5�����uۯ�M�����8c�~$��8���9��{�9�A��c?���7������φ�i�h?��=��a��������~6���Z>l$`���Q�� �8�=K�ÜF�']@[j��G��d?�5�m����i۸�kd�d?r�vn��E}��)�����j{Ǉ� u��� ?�γL����@>7�b� ���b7�o��v6|ۚ���c_�&�q���~�5�ُ���>����Cκv�q�    ���_P�o�frr�h�G��g?����~�Ww�{��k?d��+�g�W�^�I]��g?w���aJ�u�O�=��
��~�z�S~2x�8�~� ����b������>��%����a���f���{��ٸ���~6����aƀu���M�3���� g�HƱ/����=�3_���b�i��,׼�ԟ�B6��h[��tqC��|���1k��0WP7���C����Rc��s{}��=��X���si���ִ��_��l�_h[�|QC��b
����B�_������&���)�ܻ--�=�����u{X��>��X�SI٤�dж�/��>��4RW���֭�tP�ݞ�F�e ~4�#�~u.~�O��2A�����o����I5֘��h����_�B6m����5�WzqC��j�������_���}+?����p�HN��tn��U��%���/���=|��6���5o��C�G>����a7����4RW��� �Y>�=��>
�#�B)��l4뉓/8r ���b�N����~��>��������H��~d�=����;�G~QC��h
����)�������~��s�N%ⶓS�����U��]��,kDۚݐM���B6ɏ���㸘�O~<}����\�u�O�{V�.�����Q�6£�`�s(��,v��Wt�j_�'�fn��W|!�(�ֵ���}�餮�G����Ñ�����\?t|�ĵFC�GĹ�[��$�tŇ#p����Xw�|8RֽW|8r�G��	�u_�"�{��p����g�λZ>��n���i�k��?R"!b��$����)PW��9����3[�����>�ͅ��6}�,�����񫦏����|\���?�J�j�pg�����&��~����3SRԩ���H{�?��nҟ��>�y�y۩?����;!��E�� mg�σ.n��_L#u5}x`��u��7�c��+�g8�yW���� ��_�b7-~#א��/jM�N��XȦ�/	���K����~9����ᩀu�/�M�_:��[֬�Hu.l�O>��3 �c���n���֐�ٯƚ���+Z�&��u�������WSH]=^X���A�d�
��w��n�4�s��*�c����߬�א��/�i�h���=��!���_����~1���Z>b8`�����_����-�2B�G���<�(���/h,v���֘��x��N��,d��HA۪?����G�H]-AX��(Aݤ?*��w��6�RL���<�
��1-v���א�ُeM�N��.d���@�j?�����SH]-�	X���@�d?�?�y�S��O���C�W;n8��f���'�ƴ��#D׼�ԟ�B6�O�������O�C�]M!X��t��g�w�vП�Y��'sɹ-�ҕ�n7�W�Oe���w{��y���1����ּ����o�/�O}!��N�����4@ۙ�͋>�����%���i��֏@к��#��~l�8� �o� ��O�}��ͯ��C9ՂY��q{��-���ǝ�Y"g�i���_p�� ���kA���.���^�V5�[��]tu��`�t���͋�o���˂sj�AE�^~n�[��ǂn���5�}��\��h=� f]�+ s.����1.j�`L1u��D0`�uӒ8�ÂE:Q.�'7�E8{���T
�"::�@"r�ڋo~w_�`�B�4GжJ0�b�~	��RW7H� �-�TP7m��>%H/D�5d�ZSe�v�D:`�v��&�h�$2װ�-�����\�XȦ��"ж.��/z�7aM9uu�D)`�&,u�y���|�p��%R盰�W;�ď��n�$�Z����=�9朦�����Ǆ9��&�!=��0ǔSW�H�O~�w��pP�Ԅ9�i�x��B��9-G��'�H���0G-vOM�4ְ��e�hM��#��x!�LH�V�^��oB�r��Ir��MH�a��&��8*U�l�_�h.��!X�v\�<�I�LkT�Dȼf�E�7��~Q�,ٳc����|L�l;􋐧��zG��nr��g�8� ?�̚�P�#Mb*q�ƹ�#)�="Z�&
�Q�[���{��{3_��B6U�b��<4?�/v��L7u���$`�"�u�S�����!8B��P�uz��@ڲY�����6�g|D�*kL�4���m�q�-dS=�ړoF�Ը��_�:���N�Z�uk��{�NO#����C���R�1H�vt��
w��48�@���$�ּm�3N��i'I�v�������i��v�Dj���$�@��N�Hq?�������`
\�}���t�H"�]mDn���$���'I���M��m�^z]�Я�w��O�A�uk0�=I��Р�f�s�iࠉS5
ҕo��|�����$�ט�ip��r�jpVF9�48�{/����5��L]%�X�S@�T���IB4�;s. �Ob'w���.X�vt�e�b7ՃkT�D��fn�����`ж�f�.v�aM7u5�d	`�",uS=X��\|h�����'��� ���Z���d��5�}�ִm����T$w�� �v��5���v֐��`�ق5�=�`�a��ȹ(u��=92S#���l������`��/;X4 �Q�B6y�����"Yox�I�htm٤���|��=9�Xoz�0_HhX��̭���A�i��cC�;��(u�X�����&q�
��`��P,s��Ґu��]N�i)�zVX<�D�	��X�=���\�c�D4�7+���87BS\��e��j�����=.,�c�6>/,��.�R�%�p;�KJ�x˛�(SQ��aR��-DɅ��v\R����S�ɘ�����:l�Nr)��R>�oѬr��Χ���i٬�p;wQJ�J�N�i)m{��SNھ�\Z�'g]6�?<D���4<���\%-֖ݔ2>�Me��1�}e��"���f��L��#���f;��K�I�ݛw������c�U��D}�-T-T�F�M�w�菍�>�op̏���?��\���тWC�Ո���mk�+��[$���&!z�����<�a�$D���(D�gbf�bJ�z���?�!Ǹ6�P�����m]̅_M��j�j��^+��Z���V�c|*��ךS}������aǡz�|�Ѿ�U�RSA�Q*m�6J;
��w�B�Ԋ|q����V�ҞG��5��[�*��nk�X�j���SpU\}��
���N�Bu�,�&_��]#���N�\�:W�l��Do�x�#Ċc`�vU*���x���E���[^��F�/��DѮ�ɇ���)�9q�d���8��ݓ͞x;���X)S��>g=<�>ӊ��E���ǋ�ym�`���I�x��K�D��3�=__�b�	��5"��'�L�-q⧭l<=���Y3�����sw�'.ֻݲƜ�<�]rD>\;���oxӸ�����y&3y�&�ν��"�J�иQ�3�ɲ�j�#��F��̸=� l)�K�������e�y��z��7ȉ[�9I�nL)N(�.1"*n�yA-z%�bDx%�S���'F9ʧv1"3ncSPq�s�4������U�!>n���|�&��yպ�գ�j:�f���óA�_UPϖߛ,[�vQ 5n�.��8�����()2�A��#/�QC!4n����:�]VBZ�>M5?_6h"�ĭ�K�ō�������|x����5��Tj~n��K�����<�4_��-��E@Uڱ7=Y�X����ۮ4��M�6�1���������i�p�v��8�|�{2ݓ'��6~ Gn���2�v@�vb#����W���}�`�YO����G�i Rn�m��r��N�԰Jq�Б�_�H���i��t9�]��0��=��>fqc~gB��v��ȑ�޻H��+_lp#��s�pq�3>oD�|��݈����9bv���pd�Tv��,'?�1��I����U6"b��m܊A����1�v�	s��ec֕.6�qs� ����h4������"pn��FR>�<�&\���Mj��m�9:�n��hD��>�mz�� L  '2�9�jF$ͭ�,�ɫ��Z4�W��7#!unѲ�&$̭?�MH�[�2#!tn+��/�9�R�L���#��	ڞd#��]���'B��>�}?��Yܹ%NȚ[����o���&d�m_��;�� ��s�����?�1�k���W�Z�>��o����2�M7�]��5·]c�"ͽ9.D��{�A݄#��ۺ�&�+_lp#5U�Y�����V����:���=�q,vɋ���=�f{vݺ����+x�q�s��i�?��~κU��^�F��y�=��<��u�
����<��� Z���vEZ��m튴�����.�G��w�Wμ6�Ii����I�U_gٺ��ۓC+u^]�B��ym.[?w�y܈ȹ��q����;h�_C��4��jD���BOޢFX�W[Θ@�Ƣwm�������4B箽1ة]Ц�
�ۻcve�E��n�x�S��͈��۶��H��C��d��0�`�gxڈ��ۦ�Dt�/z�3��KR'R�i�*Gd���vb3�����W�� G���{䈈���wqs�]�ԄԹ�q�+�O�Ί1��g8�u�h�6-i:��k&O����6���-1���{�Nb�����"��܈عG��>D�}�N���܈Թ?��I�M'�K�<{#s�]jD���γ�<=��/�HH�{��3�~@��Ԉ���+]lP#b��s�-!b���{nS#R��B5r�0اI��C��2��%G����t�֍H�{o�as��55��ޘnD��w�����/GF�ܫ�x2F���w��s�][1��?�	v�Y��d32R���&q����:��MbF��[O�eDͽz�������h�ș{�]Q3�+[l0#r�1Z6����}3��1v��0b��pܷ�(�p��=y��1�]�oF�<�Bߌ�y���9�h}3��1��ӌ�yl}3ӕ,6����9�0����a�MWM��5��ku+㒑c�ڇ;W���Oܞ #azWՈ�y|z��3>ld˃z62��A}rD�<h�z�����A������0���/#Y�M����)�؍�L	���<W�H��ZP#`����u#2��z�/#Xͧ�2���m[Ԍhy�f5�\�b�=����	��?���6�m'�1B�!��Ɯn�������}F�<dS��;���&{ֺA�h=���5���0��!m-)��yl>��M���A�����!��~3�������o�#r籂����|5���#�?��3�]�oF�<�r��t�m�GDͣ9������ș��#m���܈�yh�52���.�ț�ns#b�W�<D1y�Z�j��F��cW� #u}���y|�0���5mތA�<�9��|�_�b��;���F�<�{by��m���c��ċ��T��kj���w����;Ֆ4�O{b��#���|�-#m�v#h��_�3�|����G�ԍ�GO#p�Nc��#�nTLY�dR�'�)�M��2B��w8#f���1���jD�<�/܈�y�f5V^�b�;��������#w��R�O���pWg��t:�i� cu������(H�G�y߂�y�v����|޷ lٶ#șG�U����E������yG���y�z�(+v^�xcZMC�r��:Y�+d^��]��;��%�9�FYI�O[b��a���y�)++n^me%�7_
-<�t�A�H��hI~R���#H���n�s<�q��J��b'�9��6
b��qW̳����]1��Fd�s��i����ssW�]�b�<O�YQ#d��]1��yҶ���t�Ⱥ?�i�T�8�.A�|�6��sл
GDͳ�+F�6��A�<��bQ��rT��9"z��S8"f��]1��y�69"y����1�BZ�XH���͈�y�.3"u��gF���Ӧ��4#���QS����fD�<�όș'o6�ɕ-6���n�y�����o���S����G3j���m��뜫F��Sv���>5"f����<���IܪFd�S��;��ynn��+]lP#b�-�6
"�����s[�[�:���7�� 3�g׌H���چA�<�2߂�y�^#���6�A���|2�9�-aW�� Fd��ZN&�˳�WP�5�m����y�S�8�¥����t5"`��Z���kD���V���2O��O�3ϾVAA�<7�
J��.6���������o�D��wE����G5b�͍J*E����jD�<w�#�g_7� b��v�<��2Oo��3O�1��7�Hŕ.6���5"^���0��yƮ�Q�8�xR�Q�af!gi��gl:�V�8ϾvE�<[��Q��3zݨ�g�1��y�^7��+_��Q�9�lYQ+�����Țg�*��\�0�B-g��S�Eet��;��y��QD�3�܈�y~��nD�<�ٍH�g��!�̽O���܈�yV�W���FWؼ��qe�W?L�H1� ���?���ymj��9�k��2om�ѕ3�f7��y_C���ymv��+_lp��z����'L������z��/�����'��?߿�F|���zs��������15V=W�(2��ߟ�Ƞ�ئjd�k�H���5��T�AՈ���U�"�^�OՈ�Wk�"�^ͷ�("��ן������U���U��Cշ7�ߪj�ԋZ�&R$ӫ�]HI/�u��"�^t<qȨ�6u*���x�T��mR52�w��j�R�Ӌ�U��z�u)�赹[HM�|�A�v��EՈ����������S�w����ۣ0L,LK��d����_�����o�4��û���~��w��_����|��û&ڻ�����/����������$����~�����7/?�~��ÿ��������	�8��p=���z��������������7��X�>̪�o��| ��/޶@q:���;1�r�Q�l$y�3|�!_��ar���
���Wk+�"_ͭ`��ھ��p�ޯ=?����ㇸ�}#2_��_{H��Uܭ�&����?�һ�/��K���V�w����r�%ۖzH��S��@6s�7��bt�9� ~ɮ/=D��Oů�&?E�����4�}��a��_z����~����(�      [   �  x�͖���0���#.	U�CL�[�S������0�ue�ch��;aŶ�F+��q �f>;��M=����z�n��F[[�~�t��ӈ�������4�0>�7}Ļ��r�[ΚJt�8�ޙ�C��K��=����E0���Tw��R��j1�ħX�&���6,�ʂoC|[��͂�&���e|j&j��UY�m��d�d|�I�o�_K�ײ|">Q����yFdD� �"��X�:@ʐ��8�G�	
�J�J��������;w.�w?���#8���N�׮5Ur4��u>�290o�:i��P�������*�-��U��v���q��G�_�.Eyy�Aлl�j�HX�b�^G���Ԗ<հ㤒��S��t��׊/%^W���%����x�~]:��ao�J���]�O���Ӷ�b���?G^��x�wތ�������Z����C�������y�"�gZ�aK%?l���Ӳa����S��0���!�      Y      x������ � �      _     x�}VI�$9<��t�����0.)�8�JK+!�AH�W;>3յ��DM�5���uپ��;�eZxF����_[���p�
�
���D�R��b%�=6n�h�Dͬ�:/`��2�Μ�XM}�U�1F�\Cs���6c�/a��}�d�u�U'J��T�ɻ"����{	��GG�$8D�vL��U��Y�����B�p<����� �8&82ED� � ��pr��������J��`�.�%��o�@BPs�a� ��U��@��@��2�B�q�`]
O
��مC�U P��E��({7<�R#�@�e`�?@�qR?��}>(���A�C���i4�e0ni�^$v�u	�--�K�IvX7桳�I��(FM"�V��C:'�ʚ��9�bL���3��yv\T���23!��Y��aF{���d�0Y<�I�s����d�0���ٙ|��߆2 ��L�����}Ѭ�D�%3c��?�r#��P�X��<ɯ��D���*�Z^����@��?�
�x�+��A�P�bav��!:���][+��"�Ɀ�� *����(��}���zK�A��ʦ��	φ�B��C+�bN��R��""H�L�G� ^S�qFc�+�r��_B����Ѐ^��\�(LJ�|0)��PbR= ʢ� A����Ϧ|0)����b��J�L*��3�!�
�A
�e�Q����kr���`r*�����S�նV�o��,3���诤��g=Y�1p�q��=x���`à����ħ5��D�<¥�z �L� ���γ=�1XA�)�V+��FTN��_�0��c���j��*V�W���r)c�b���&w����z�A�bKb�3�������9�ɇ��F@�\L?�JE� K���T����ZaE�F����r�=:F'8��$`Cu��Jxm4/�)�5��F7����b�e��<�>c��\:�_�P���aki ����VE�,�mR�٧������`���5{z4� �]Գ6�[d`�k�1��,Kx�G���?�Z�_<�      W   �  x��[ˎ캭�_�8�ė����rw� ��i����\蒪mTo4��e�2E..Rli��V�����������V�5��hեoR��V�7�?U}��92,��K���Y����j�(-�T��y�6	y�V#�7h��e��]��Em�a�=%�d��nkv��.i�n5����Wſ�û��b^\��?���TH<j�٫�KjG��Rٱ�jb�dT�=k��ͮ���X�,��%v��ګ�4o;vl=꬘'ԏh�}�K֏��v����r�f1����2�<v�ëp�u�fN5��7{==��I>��oDۑEj_�`���n�56��jIX6񟷾��=D���ӛ7҃���u!���a�R��WW��k�Kl�G�#�{ۦ�+�K6^`c�Տ��}�3,YMq��6{Ci���O8ܲ_c�I�ٵ�?J� w�סX�*��WU�*990v��a0�|_���+�̊�޷|�o+=�w�XB։���b?=j�]ˆ��4�qD0%MڬK�
	��l}2s�����"���`��
z��������W���]4pv�[�(x�!��b) 6,P��#@�'�f�0��;ps;`�� SgL�{zX�~E�`�|'����ٶkݴ����x/�h]��{��X�GUyݶ�Z���'o�DG��8Le̛b��s�Nn-֠���!���^�����F�DL ��h��ab�^G�mM�j�oIc,1�*���M-@�C�f,�D$� �����S,�AU�Ͱ�/y
�f��u1V�� �ܱZ��E.�WJ�������XBju8N�=�Lm�j`i�@D@�U��� ���R�X��Zw�M�'��W#a�]��q/E�
`�)s�H-p�� �֪��$�(M"�*2@�@�@��c�Y'>`_�&#ְw�5�e��"�S6l�RMD)��>{���ാY�� �=�c�<۩� >���|���5F�VK��Hr��P _�à������3�#�pn��zCm��u��"���Μ9!^G�d��&�G���k�%��$)@`H}�L0s�?�0��h� Zd�K���!��S-��缜��:��`Rj��W�N�9jn(�"��	��7X2Cj��������	���+�Ik+�(�����7W&��B%7�H�n$�$p��[fL�a.k��k�w�M?g	��r ��h�Ĭ4T@m��\���B��%�Q���-@$�%��{4u$N�Y� ���]�\�%�x� ꚛ��lt NR�ʎ/8*��{���Y<׆��� Z�)F��.9�P۵o����Z��..x����t���K.@ u�=���4Sv�5�c��6���W{8�?�����u��v��}1��*�e�����{C �V�yc�<��K-&�����f?�M1�M�1��$��b�*\7�U e��j1W�\&�0���*T���ׯ�����_S x!�;0	��_��f?HzACf	[�~mt7�.H�{�-^�Q�m	�\ʓ��A���M��B΁Ĺ�o���X�"ԥz���}���65j��H�R��MD�Z�����W�َ��_k��ڒ�����)Y����9|�Z>�*%g��(�@qc���V�$ϱᓻ�8��<"xb@sdH��K�W�@F�-7������u���g���f?@�hݾ`ra�(�D�l�yS**Y	�d�ިtA���g�ϱB��B^9-)����J�^7���Q�Y�B���AJ��Bv����g�3>Z��/ `�pG �n.�Y:Q��C�M�Iq���-�Р��a[|��+ �|,�H�8����F���N��g�hgS������>5ҤFX߁��|Exh�#�,̙Y
��*�;/Y����#->Z}pع6�
>�g���Zow�*���Y��"����j�殳y�42̐{��y�G�Ok�Q��6wE�rlW���m1���gu�#����cs���N+$��t�#���yA�A�n�������d�>Un����%p�FY�=�u�\Fb�3!z�cI|o�ʧ[�qa�Q_ز��0�jou�ϕ���>W8��A�'j�a����2,���r��
��dщ��oK��}ZB� �y m��U}�+�\{��ǩ�����*>q��4��n>�;����/@qQq�e�	�T�� _��`�߳b����»��F:�޺��<j���o��VN�Ja_�S��++��).���h�P��܁����c� xm5vcp�k���]��<a��y<��Z�����a��y��������ypĻL�����<���� �c����í��6�z<O�7���- �ϣ��{ޏ�r�o��=� �ȋ2��-���Q>ޑw���t��0&H��w��䫞u΍ 2D��S@�_n$��Qշ�+���Zɀ��O	ƀ_
�R���7��-)�=1
��v��x�Qp�<��* ��1�ĸޓ� ��F�������l \	#!a��*9܍ ��A�O�3��p�8�����-HXr�+�ӮOZY\�op�ҵ a�^��R�k	wv�O�����a�����4T2�\I(�B�MN�άp#��֢�0\y�r}�:Zخ��D�ڍD�?`�)!����� +�tB�;\9�}q�'�V^V\9����v���k��3?'�\�i���	�*P$���
�Z��T9������&������o����&NQ�	�o����^��	�mx�S���Z@Q^v���ϕ&���J~9ލ jL�+�[����&�>x�p#��	�	j��޵ ��������;P�<����@洓���J"�8���)1��Oz?�G�Aso$X�p����F�7-��C1n��O�4���T^7��p
a4�/�
�i?u$	s�n�۳g���zv��|����m��c�x�5���ǒ�⭄���y�F8V橭�σ��v��r7����5�yŪ((j�j7ƫ����A�����J�6¬Kl\���\F��K��'����.�/��jlٌq�s��|��`����n���V؊���}�g|��X�Z�u�yev����e�r�e1b23%6�J�zaZ�R�^�2,t�A�k�9���}�&i���ueL��s3�8��� Ӏ�S{�Tcd���LS���X��ұ���5�xdw��y��=����g
�e����Nf��)K %{��;\�qN�^��Y��N5�T�T�Q�\����׭�0�Z���3d���5 m��9<�á��s�ys��#j��9Jdn�k����#G�4��gw1"����v�)J����>����1�>�u�A|689o��e�|�$'?�t�o#�mtx�$� k]� <�H�d�3F�Z�|����xp������!��xcx�JD�p����I~�� tӋ4=$�U���t�-�>�8�^	٬Җ%��9�q��� �ƶ����\ĕ���q��������/�QèZɧ5�S��	ex��%ޘ?�/����.1�gAu
<��'_`P�>�圻�F4jW�%���#�Ս����m1�Z�����z��#��4V4��E�9yu��Y[D>�����q�E����W�M�L���/����ހR�����q�����^�c���������/�yA=�VAA$�Ƀ!�@�4���L�-b��y֦��H��Kk�~.�n9S�Gsj
���E7XA?Q���p(@� ���
1��t'T+�ȿ�F]��C�� �o$|��69�I�ANⶍC���|��P�,��pV��(�#s/���7����<Ͷ\��'妠:8���Ƴ�ɑ>%����D�fN�����S&>n��6*�Mto�G���8w��͠\��'k��"	O!����H�<`^�@M�\Ƿjxq�YN#�c��$f�CGߥj<;FkZ+�ui�]yv�P���wv��Vj����TbIf�
(.R�D2��g�+o�߾wjИ�Jy��㒐�DB���c����؛      Q      x�3�44�4�3�3������ �J      U    c           0    0    ENCODING    ENCODING        SET client_encoding = 'UTF8';
                      false            d           0    0 
   STDSTRINGS 
   STDSTRINGS     (   SET standard_conforming_strings = 'on';
                      false            e           0    0 
   SEARCHPATH 
   SEARCHPATH     8   SELECT pg_catalog.set_config('search_path', '', false);
                      false            f           1262    16409    optuna    DATABASE     x   CREATE DATABASE optuna WITH TEMPLATE = template0 ENCODING = 'UTF8' LC_COLLATE = 'en_US.UTF-8' LC_CTYPE = 'en_US.UTF-8';
    DROP DATABASE optuna;
                hannan    false                        2615    2200    public    SCHEMA        CREATE SCHEMA public;
    DROP SCHEMA public;
                hannan    false            g           0    0    SCHEMA public    COMMENT     6   COMMENT ON SCHEMA public IS 'standard public schema';
                   hannan    false    3            h           0    0    SCHEMA public    ACL     �   REVOKE ALL ON SCHEMA public FROM rdsadmin;
REVOKE ALL ON SCHEMA public FROM PUBLIC;
GRANT ALL ON SCHEMA public TO hannan;
GRANT ALL ON SCHEMA public TO PUBLIC;
                   hannan    false    3            )           1247    16411    studydirection    TYPE     ]   CREATE TYPE public.studydirection AS ENUM (
    'NOT_SET',
    'MINIMIZE',
    'MAXIMIZE'
);
 !   DROP TYPE public.studydirection;
       public          hannan    false    3            ,           1247    16418 
   trialstate    TYPE     r   CREATE TYPE public.trialstate AS ENUM (
    'RUNNING',
    'COMPLETE',
    'PRUNED',
    'FAIL',
    'WAITING'
);
    DROP TYPE public.trialstate;
       public          hannan    false    3            �            1259    16565    alembic_version    TABLE     X   CREATE TABLE public.alembic_version (
    version_num character varying(32) NOT NULL
);
 #   DROP TABLE public.alembic_version;
       public         heap    hannan    false    3            �            1259    16431    studies    TABLE     �   CREATE TABLE public.studies (
    study_id integer NOT NULL,
    study_name character varying(512) NOT NULL,
    direction public.studydirection NOT NULL
);
    DROP TABLE public.studies;
       public         heap    hannan    false    3    553            �            1259    16429    studies_study_id_seq    SEQUENCE     �   CREATE SEQUENCE public.studies_study_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 +   DROP SEQUENCE public.studies_study_id_seq;
       public          hannan    false    203    3            i           0    0    studies_study_id_seq    SEQUENCE OWNED BY     M   ALTER SEQUENCE public.studies_study_id_seq OWNED BY public.studies.study_id;
          public          hannan    false    202            �            1259    16467    study_system_attributes    TABLE     �   CREATE TABLE public.study_system_attributes (
    study_system_attribute_id integer NOT NULL,
    study_id integer,
    key character varying(512),
    value_json character varying(2048)
);
 +   DROP TABLE public.study_system_attributes;
       public         heap    hannan    false    3            �            1259    16465 5   study_system_attributes_study_system_attribute_id_seq    SEQUENCE     �   CREATE SEQUENCE public.study_system_attributes_study_system_attribute_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 L   DROP SEQUENCE public.study_system_attributes_study_system_attribute_id_seq;
       public          hannan    false    3    208            j           0    0 5   study_system_attributes_study_system_attribute_id_seq    SEQUENCE OWNED BY     �   ALTER SEQUENCE public.study_system_attributes_study_system_attribute_id_seq OWNED BY public.study_system_attributes.study_system_attribute_id;
          public          hannan    false    207            �            1259    16449    study_user_attributes    TABLE     �   CREATE TABLE public.study_user_attributes (
    study_user_attribute_id integer NOT NULL,
    study_id integer,
    key character varying(512),
    value_json character varying(2048)
);
 )   DROP TABLE public.study_user_attributes;
       public         heap    hannan    false    3            �            1259    16447 1   study_user_attributes_study_user_attribute_id_seq    SEQUENCE     �   CREATE SEQUENCE public.study_user_attributes_study_user_attribute_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 H   DROP SEQUENCE public.study_user_attributes_study_user_attribute_id_seq;
       public          hannan    false    3    206            k           0    0 1   study_user_attributes_study_user_attribute_id_seq    SEQUENCE OWNED BY     �   ALTER SEQUENCE public.study_user_attributes_study_user_attribute_id_seq OWNED BY public.study_user_attributes.study_user_attribute_id;
          public          hannan    false    205            �            1259    16534    trial_params    TABLE     �   CREATE TABLE public.trial_params (
    param_id integer NOT NULL,
    trial_id integer,
    param_name character varying(512),
    param_value double precision,
    distribution_json character varying(2048)
);
     DROP TABLE public.trial_params;
       public         heap    hannan    false    3            �            1259    16532    trial_params_param_id_seq    SEQUENCE     �   CREATE SEQUENCE public.trial_params_param_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 0   DROP SEQUENCE public.trial_params_param_id_seq;
       public          hannan    false    3    216            l           0    0    trial_params_param_id_seq    SEQUENCE OWNED BY     W   ALTER SEQUENCE public.trial_params_param_id_seq OWNED BY public.trial_params.param_id;
          public          hannan    false    215            �            1259    16516    trial_system_attributes    TABLE     �   CREATE TABLE public.trial_system_attributes (
    trial_system_attribute_id integer NOT NULL,
    trial_id integer,
    key character varying(512),
    value_json character varying(2048)
);
 +   DROP TABLE public.trial_system_attributes;
       public         heap    hannan    false    3            �            1259    16514 5   trial_system_attributes_trial_system_attribute_id_seq    SEQUENCE     �   CREATE SEQUENCE public.trial_system_attributes_trial_system_attribute_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 L   DROP SEQUENCE public.trial_system_attributes_trial_system_attribute_id_seq;
       public          hannan    false    214    3            m           0    0 5   trial_system_attributes_trial_system_attribute_id_seq    SEQUENCE OWNED BY     �   ALTER SEQUENCE public.trial_system_attributes_trial_system_attribute_id_seq OWNED BY public.trial_system_attributes.trial_system_attribute_id;
          public          hannan    false    213            �            1259    16498    trial_user_attributes    TABLE     �   CREATE TABLE public.trial_user_attributes (
    trial_user_attribute_id integer NOT NULL,
    trial_id integer,
    key character varying(512),
    value_json character varying(2048)
);
 )   DROP TABLE public.trial_user_attributes;
       public         heap    hannan    false    3            �            1259    16496 1   trial_user_attributes_trial_user_attribute_id_seq    SEQUENCE     �   CREATE SEQUENCE public.trial_user_attributes_trial_user_attribute_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 H   DROP SEQUENCE public.trial_user_attributes_trial_user_attribute_id_seq;
       public          hannan    false    3    212            n           0    0 1   trial_user_attributes_trial_user_attribute_id_seq    SEQUENCE OWNED BY     �   ALTER SEQUENCE public.trial_user_attributes_trial_user_attribute_id_seq OWNED BY public.trial_user_attributes.trial_user_attribute_id;
          public          hannan    false    211            �            1259    16552    trial_values    TABLE     �   CREATE TABLE public.trial_values (
    trial_value_id integer NOT NULL,
    trial_id integer,
    step integer,
    value double precision
);
     DROP TABLE public.trial_values;
       public         heap    hannan    false    3            �            1259    16550    trial_values_trial_value_id_seq    SEQUENCE     �   CREATE SEQUENCE public.trial_values_trial_value_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 6   DROP SEQUENCE public.trial_values_trial_value_id_seq;
       public          hannan    false    3    218            o           0    0    trial_values_trial_value_id_seq    SEQUENCE OWNED BY     c   ALTER SEQUENCE public.trial_values_trial_value_id_seq OWNED BY public.trial_values.trial_value_id;
          public          hannan    false    217            �            1259    16485    trials    TABLE       CREATE TABLE public.trials (
    trial_id integer NOT NULL,
    number integer,
    study_id integer,
    state public.trialstate NOT NULL,
    value double precision,
    datetime_start timestamp without time zone,
    datetime_complete timestamp without time zone
);
    DROP TABLE public.trials;
       public         heap    hannan    false    3    556            �            1259    16483    trials_trial_id_seq    SEQUENCE     �   CREATE SEQUENCE public.trials_trial_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 *   DROP SEQUENCE public.trials_trial_id_seq;
       public          hannan    false    3    210            p           0    0    trials_trial_id_seq    SEQUENCE OWNED BY     K   ALTER SEQUENCE public.trials_trial_id_seq OWNED BY public.trials.trial_id;
          public          hannan    false    209            �            1259    16441    version_info    TABLE     �   CREATE TABLE public.version_info (
    version_info_id integer NOT NULL,
    schema_version integer,
    library_version character varying(256),
    CONSTRAINT version_info_version_info_id_check CHECK ((version_info_id = 1))
);
     DROP TABLE public.version_info;
       public         heap    hannan    false    3            �           2604    16434    studies study_id    DEFAULT     t   ALTER TABLE ONLY public.studies ALTER COLUMN study_id SET DEFAULT nextval('public.studies_study_id_seq'::regclass);
 ?   ALTER TABLE public.studies ALTER COLUMN study_id DROP DEFAULT;
       public          hannan    false    203    202    203            �           2604    16470 1   study_system_attributes study_system_attribute_id    DEFAULT     �   ALTER TABLE ONLY public.study_system_attributes ALTER COLUMN study_system_attribute_id SET DEFAULT nextval('public.study_system_attributes_study_system_attribute_id_seq'::regclass);
 `   ALTER TABLE public.study_system_attributes ALTER COLUMN study_system_attribute_id DROP DEFAULT;
       public          hannan    false    207    208    208            �           2604    16452 -   study_user_attributes study_user_attribute_id    DEFAULT     �   ALTER TABLE ONLY public.study_user_attributes ALTER COLUMN study_user_attribute_id SET DEFAULT nextval('public.study_user_attributes_study_user_attribute_id_seq'::regclass);
 \   ALTER TABLE public.study_user_attributes ALTER COLUMN study_user_attribute_id DROP DEFAULT;
       public          hannan    false    206    205    206            �           2604    16537    trial_params param_id    DEFAULT     ~   ALTER TABLE ONLY public.trial_params ALTER COLUMN param_id SET DEFAULT nextval('public.trial_params_param_id_seq'::regclass);
 D   ALTER TABLE public.trial_params ALTER COLUMN param_id DROP DEFAULT;
       public          hannan    false    216    215    216            �           2604    16519 1   trial_system_attributes trial_system_attribute_id    DEFAULT     �   ALTER TABLE ONLY public.trial_system_attributes ALTER COLUMN trial_system_attribute_id SET DEFAULT nextval('public.trial_system_attributes_trial_system_attribute_id_seq'::regclass);
 `   ALTER TABLE public.trial_system_attributes ALTER COLUMN trial_system_attribute_id DROP DEFAULT;
       public          hannan    false    214    213    214            �           2604    16501 -   trial_user_attributes trial_user_attribute_id    DEFAULT     �   ALTER TABLE ONLY public.trial_user_attributes ALTER COLUMN trial_user_attribute_id SET DEFAULT nextval('public.trial_user_attributes_trial_user_attribute_id_seq'::regclass);
 \   ALTER TABLE public.trial_user_attributes ALTER COLUMN trial_user_attribute_id DROP DEFAULT;
       public          hannan    false    211    212    212            �           2604    16555    trial_values trial_value_id    DEFAULT     �   ALTER TABLE ONLY public.trial_values ALTER COLUMN trial_value_id SET DEFAULT nextval('public.trial_values_trial_value_id_seq'::regclass);
 J   ALTER TABLE public.trial_values ALTER COLUMN trial_value_id DROP DEFAULT;
       public          hannan    false    218    217    218            �           2604    16488    trials trial_id    DEFAULT     r   ALTER TABLE ONLY public.trials ALTER COLUMN trial_id SET DEFAULT nextval('public.trials_trial_id_seq'::regclass);
 >   ALTER TABLE public.trials ALTER COLUMN trial_id DROP DEFAULT;
       public          hannan    false    209    210    210            `          0    16565    alembic_version 
   TABLE DATA           6   COPY public.alembic_version (version_num) FROM stdin;
    public          hannan    false    219            P          0    16431    studies 
   TABLE DATA           B   COPY public.studies (study_id, study_name, direction) FROM stdin;
    public          hannan    false    203            U          0    16467    study_system_attributes 
   TABLE DATA           g   COPY public.study_system_attributes (study_system_attribute_id, study_id, key, value_json) FROM stdin;
    public          hannan    false    208           S          0    16449    study_user_attributes 
   TABLE DATA           c   COPY public.study_user_attributes (study_user_attribute_id, study_id, key, value_json) FROM stdin;
    public          hannan    false    206           ]          0    16534    trial_params 
   TABLE DATA           f   COPY public.trial_params (param_id, trial_id, param_name, param_value, distribution_json) FROM stdin;
    public          hannan    false    216           [          0    16516    trial_system_attributes 
   TABLE DATA           g   COPY public.trial_system_attributes (trial_system_attribute_id, trial_id, key, value_json) FROM stdin;
    public          hannan    false    214   `       Y          0    16498    trial_user_attributes 
   TABLE DATA           c   COPY public.trial_user_attributes (trial_user_attribute_id, trial_id, key, value_json) FROM stdin;
    public          hannan    false    212          _          0    16552    trial_values 
   TABLE DATA           M   COPY public.trial_values (trial_value_id, trial_id, step, value) FROM stdin;
    public          hannan    false    218           W          0    16485    trials 
   TABLE DATA           m   COPY public.trials (trial_id, number, study_id, state, value, datetime_start, datetime_complete) FROM stdin;
    public          hannan    false    210   (       Q          0    16441    version_info 
   TABLE DATA           X   COPY public.version_info (version_info_id, schema_version, library_version) FROM stdin;
    public          hannan    false    204   �       q           0    0    studies_study_id_seq    SEQUENCE SET     C   SELECT pg_catalog.setval('public.studies_study_id_seq', 24, true);
          public          hannan    false    202            r           0    0 5   study_system_attributes_study_system_attribute_id_seq    SEQUENCE SET     d   SELECT pg_catalog.setval('public.study_system_attributes_study_system_attribute_id_seq', 1, false);
          public          hannan    false    207            s           0    0 1   study_user_attributes_study_user_attribute_id_seq    SEQUENCE SET     `   SELECT pg_catalog.setval('public.study_user_attributes_study_user_attribute_id_seq', 1, false);
          public          hannan    false    205            t           0    0    trial_params_param_id_seq    SEQUENCE SET     J   SELECT pg_catalog.setval('public.trial_params_param_id_seq', 1481, true);
          public          hannan    false    215            u           0    0 5   trial_system_attributes_trial_system_attribute_id_seq    SEQUENCE SET     d   SELECT pg_catalog.setval('public.trial_system_attributes_trial_system_attribute_id_seq', 25, true);
          public          hannan    false    213            v           0    0 1   trial_user_attributes_trial_user_attribute_id_seq    SEQUENCE SET     `   SELECT pg_catalog.setval('public.trial_user_attributes_trial_user_attribute_id_seq', 1, false);
          public          hannan    false    211            w           0    0    trial_values_trial_value_id_seq    SEQUENCE SET     O   SELECT pg_catalog.setval('public.trial_values_trial_value_id_seq', 100, true);
          public          hannan    false    217            x           0    0    trials_trial_id_seq    SEQUENCE SET     C   SELECT pg_catalog.setval('public.trials_trial_id_seq', 193, true);
          public          hannan    false    209            �           2606    16569 #   alembic_version alembic_version_pkc 
   CONSTRAINT     j   ALTER TABLE ONLY public.alembic_version
    ADD CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num);
 M   ALTER TABLE ONLY public.alembic_version DROP CONSTRAINT alembic_version_pkc;
       public            hannan    false    219            �           2606    16439    studies studies_pkey 
   CONSTRAINT     X   ALTER TABLE ONLY public.studies
    ADD CONSTRAINT studies_pkey PRIMARY KEY (study_id);
 >   ALTER TABLE ONLY public.studies DROP CONSTRAINT studies_pkey;
       public            hannan    false    203            �           2606    16475 4   study_system_attributes study_system_attributes_pkey 
   CONSTRAINT     �   ALTER TABLE ONLY public.study_system_attributes
    ADD CONSTRAINT study_system_attributes_pkey PRIMARY KEY (study_system_attribute_id);
 ^   ALTER TABLE ONLY public.study_system_attributes DROP CONSTRAINT study_system_attributes_pkey;
       public            hannan    false    208            �           2606    16477 @   study_system_attributes study_system_attributes_study_id_key_key 
   CONSTRAINT     �   ALTER TABLE ONLY public.study_system_attributes
    ADD CONSTRAINT study_system_attributes_study_id_key_key UNIQUE (study_id, key);
 j   ALTER TABLE ONLY public.study_system_attributes DROP CONSTRAINT study_system_attributes_study_id_key_key;
       public            hannan    false    208    208            �           2606    16457 0   study_user_attributes study_user_attributes_pkey 
   CONSTRAINT     �   ALTER TABLE ONLY public.study_user_attributes
    ADD CONSTRAINT study_user_attributes_pkey PRIMARY KEY (study_user_attribute_id);
 Z   ALTER TABLE ONLY public.study_user_attributes DROP CONSTRAINT study_user_attributes_pkey;
       public            hannan    false    206            �           2606    16459 <   study_user_attributes study_user_attributes_study_id_key_key 
   CONSTRAINT     �   ALTER TABLE ONLY public.study_user_attributes
    ADD CONSTRAINT study_user_attributes_study_id_key_key UNIQUE (study_id, key);
 f   ALTER TABLE ONLY public.study_user_attributes DROP CONSTRAINT study_user_attributes_study_id_key_key;
       public            hannan    false    206    206            �           2606    16542    trial_params trial_params_pkey 
   CONSTRAINT     b   ALTER TABLE ONLY public.trial_params
    ADD CONSTRAINT trial_params_pkey PRIMARY KEY (param_id);
 H   ALTER TABLE ONLY public.trial_params DROP CONSTRAINT trial_params_pkey;
       public            hannan    false    216            �           2606    16544 1   trial_params trial_params_trial_id_param_name_key 
   CONSTRAINT     |   ALTER TABLE ONLY public.trial_params
    ADD CONSTRAINT trial_params_trial_id_param_name_key UNIQUE (trial_id, param_name);
 [   ALTER TABLE ONLY public.trial_params DROP CONSTRAINT trial_params_trial_id_param_name_key;
       public            hannan    false    216    216            �           2606    16524 4   trial_system_attributes trial_system_attributes_pkey 
   CONSTRAINT     �   ALTER TABLE ONLY public.trial_system_attributes
    ADD CONSTRAINT trial_system_attributes_pkey PRIMARY KEY (trial_system_attribute_id);
 ^   ALTER TABLE ONLY public.trial_system_attributes DROP CONSTRAINT trial_system_attributes_pkey;
       public            hannan    false    214            �           2606    16526 @   trial_system_attributes trial_system_attributes_trial_id_key_key 
   CONSTRAINT     �   ALTER TABLE ONLY public.trial_system_attributes
    ADD CONSTRAINT trial_system_attributes_trial_id_key_key UNIQUE (trial_id, key);
 j   ALTER TABLE ONLY public.trial_system_attributes DROP CONSTRAINT trial_system_attributes_trial_id_key_key;
       public            hannan    false    214    214            �           2606    16506 0   trial_user_attributes trial_user_attributes_pkey 
   CONSTRAINT     �   ALTER TABLE ONLY public.trial_user_attributes
    ADD CONSTRAINT trial_user_attributes_pkey PRIMARY KEY (trial_user_attribute_id);
 Z   ALTER TABLE ONLY public.trial_user_attributes DROP CONSTRAINT trial_user_attributes_pkey;
       public            hannan    false    212            �           2606    16508 <   trial_user_attributes trial_user_attributes_trial_id_key_key 
   CONSTRAINT     �   ALTER TABLE ONLY public.trial_user_attributes
    ADD CONSTRAINT trial_user_attributes_trial_id_key_key UNIQUE (trial_id, key);
 f   ALTER TABLE ONLY public.trial_user_attributes DROP CONSTRAINT trial_user_attributes_trial_id_key_key;
       public            hannan    false    212    212            �           2606    16557    trial_values trial_values_pkey 
   CONSTRAINT     h   ALTER TABLE ONLY public.trial_values
    ADD CONSTRAINT trial_values_pkey PRIMARY KEY (trial_value_id);
 H   ALTER TABLE ONLY public.trial_values DROP CONSTRAINT trial_values_pkey;
       public            hannan    false    218            �           2606    16559 +   trial_values trial_values_trial_id_step_key 
   CONSTRAINT     p   ALTER TABLE ONLY public.trial_values
    ADD CONSTRAINT trial_values_trial_id_step_key UNIQUE (trial_id, step);
 U   ALTER TABLE ONLY public.trial_values DROP CONSTRAINT trial_values_trial_id_step_key;
       public            hannan    false    218    218            �           2606    16490    trials trials_pkey 
   CONSTRAINT     V   ALTER TABLE ONLY public.trials
    ADD CONSTRAINT trials_pkey PRIMARY KEY (trial_id);
 <   ALTER TABLE ONLY public.trials DROP CONSTRAINT trials_pkey;
       public            hannan    false    210            �           2606    16446    version_info version_info_pkey 
   CONSTRAINT     i   ALTER TABLE ONLY public.version_info
    ADD CONSTRAINT version_info_pkey PRIMARY KEY (version_info_id);
 H   ALTER TABLE ONLY public.version_info DROP CONSTRAINT version_info_pkey;
       public            hannan    false    204            �           1259    16440    ix_studies_study_name    INDEX     V   CREATE UNIQUE INDEX ix_studies_study_name ON public.studies USING btree (study_name);
 )   DROP INDEX public.ix_studies_study_name;
       public            hannan    false    203            �           2606    16478 =   study_system_attributes study_system_attributes_study_id_fkey    FK CONSTRAINT     �   ALTER TABLE ONLY public.study_system_attributes
    ADD CONSTRAINT study_system_attributes_study_id_fkey FOREIGN KEY (study_id) REFERENCES public.studies(study_id);
 g   ALTER TABLE ONLY public.study_system_attributes DROP CONSTRAINT study_system_attributes_study_id_fkey;
       public          hannan    false    208    203    3755            �           2606    16460 9   study_user_attributes study_user_attributes_study_id_fkey    FK CONSTRAINT     �   ALTER TABLE ONLY public.study_user_attributes
    ADD CONSTRAINT study_user_attributes_study_id_fkey FOREIGN KEY (study_id) REFERENCES public.studies(study_id);
 c   ALTER TABLE ONLY public.study_user_attributes DROP CONSTRAINT study_user_attributes_study_id_fkey;
       public          hannan    false    203    3755    206            �           2606    16545 '   trial_params trial_params_trial_id_fkey    FK CONSTRAINT     �   ALTER TABLE ONLY public.trial_params
    ADD CONSTRAINT trial_params_trial_id_fkey FOREIGN KEY (trial_id) REFERENCES public.trials(trial_id);
 Q   ALTER TABLE ONLY public.trial_params DROP CONSTRAINT trial_params_trial_id_fkey;
       public          hannan    false    210    3767    216            �           2606    16527 =   trial_system_attributes trial_system_attributes_trial_id_fkey    FK CONSTRAINT     �   ALTER TABLE ONLY public.trial_system_attributes
    ADD CONSTRAINT trial_system_attributes_trial_id_fkey FOREIGN KEY (trial_id) REFERENCES public.trials(trial_id);
 g   ALTER TABLE ONLY public.trial_system_attributes DROP CONSTRAINT trial_system_attributes_trial_id_fkey;
       public          hannan    false    214    210    3767            �           2606    16509 9   trial_user_attributes trial_user_attributes_trial_id_fkey    FK CONSTRAINT     �   ALTER TABLE ONLY public.trial_user_attributes
    ADD CONSTRAINT trial_user_attributes_trial_id_fkey FOREIGN KEY (trial_id) REFERENCES public.trials(trial_id);
 c   ALTER TABLE ONLY public.trial_user_attributes DROP CONSTRAINT trial_user_attributes_trial_id_fkey;
       public          hannan    false    3767    212    210            �           2606    16560 '   trial_values trial_values_trial_id_fkey    FK CONSTRAINT     �   ALTER TABLE ONLY public.trial_values
    ADD CONSTRAINT trial_values_trial_id_fkey FOREIGN KEY (trial_id) REFERENCES public.trials(trial_id);
 Q   ALTER TABLE ONLY public.trial_values DROP CONSTRAINT trial_values_trial_id_fkey;
       public          hannan    false    3767    210    218            �           2606    16491    trials trials_study_id_fkey    FK CONSTRAINT     �   ALTER TABLE ONLY public.trials
    ADD CONSTRAINT trials_study_id_fkey FOREIGN KEY (study_id) REFERENCES public.studies(study_id);
 E   ALTER TABLE ONLY public.trials DROP CONSTRAINT trials_study_id_fkey;
       public          hannan    false    210    3755    203           