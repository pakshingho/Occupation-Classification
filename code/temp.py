#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 23:44:03 2019

@author: Pak Shing Ho

Analyst, decision, research, analyse, analyze, analysis, analy, engineer

Old:
Fire Investigators, Reporters and Correspondents, Orthodontists

New (based on LR):
    False negative:
Police Detectives, Criminal Investigators and Special Agents, Coroners,
Reporters and Correspondents, Fire Investigators, Historians,
Many Inspectors, Technicians and Speicialists

Tellers, Graduate Teaching Assistants, Travel Agents, Graphic Designers, Pilots, Ship, Bookkeeping, Accounting, and Auditing Clerks (most contradictory between ML methods)
Poets, Lyricists and Creative Writers (LR, LDA say no, other yes )

False positive:
Interviewers, Except Eligibility and Loan, Order Clerks (most of the clerks),
Travel Agents
"""



import pandas as pd

df = pd.read_excel('db_24_1_excel/Occupation Data.xlsx')

"""
analyst = df.loc[df['Title'].str.contains('Analyst'), 'Title']
engineer = df.loc[df['Title'].str.contains('Engineer'), 'Title']
research = df.loc[df['Description'].str.contains('research'), 'Title']
analyze = df.loc[df['Description'].str.contains('analyze'), 'Title']
analysis = df.loc[df['Description'].str.contains('analysis'), 'Title']
"""

# list of hand-labeled analytical occupations (45)
analytical = list(set(
             ['Logistics Analysts',
              'Management Analysts',
              'Market Research Analysts and Marketing Specialists',
              'Budget Analysts',
              'Credit Analysts',
              'Financial Analysts',
              'Financial Quantitative Analysts',
              'Fraud Examiners, Investigators and Analysts',
              'Computer Systems Analysts',
              'Information Security Analysts',
              'Business Intelligence Analysts',
              'Operations Research Analysts',
              'Climate Change Analysts',
              'Quality Control Analysts',
              'Intelligence Analysts',
              'Risk Management Specialists',
              'Political Scientists',
              'Computer and Information Research Scientists',
              'Computer Network Architects',
              'Search Marketing Strategists',
              'Mathematicians',
              'Electrical Engineering Technologists',
              'Animal Scientists',
              'Food Scientists and Technologists',
              'Soil and Plant Scientists',
              'Biochemists and Biophysicists',
              'Zoologists and Wildlife Biologists',
              'Bioinformatics Scientists',
              'Medical Scientists, Except Epidemiologists',
              'Physicists',
              'Economists',
              'Computer Science Teachers, Postsecondary',
              'Mathematical Science Teachers, Postsecondary',
              'Engineering Teachers, Postsecondary',
              'Physics Teachers, Postsecondary',
              'Aerospace Engineers',
              'Biomedical Engineers',
              'Civil Engineers',
              'Electrical Engineers',
              'Mechanical Engineers',
              'Chemical Engineers',
              'Nuclear Engineers',
              'Actuaries',
              'Auditors',
              'Software Developers, Applications'
              ]))


# list of hand-labeled non-analytical occupations (45)
non_analytical = list(set(
                 ['Bus Drivers, Transit and Intercity',
                  'Bus Drivers, School or Special Client',
                  'Driver/Sales Workers',
                  'Heavy and Tractor-Trailer Truck Drivers',
                  'Light Truck or Delivery Services Drivers',
                  'Taxi Drivers and Chauffeurs',
                  'Janitors and Cleaners, Except Maids and Housekeeping Cleaners',
                  'Maids and Housekeeping Cleaners',
                  'Cooks, Fast Food',
                  'Cooks, Institution and Cafeteria',
                  'Cooks, Private Household',
                  'Cooks, Restaurant',
                  'Food Preparation Workers',
                  'Bartenders',
                  'Waiters and Waitresses',
                  'Food Servers, Nonrestaurant',
                  'Dining Room and Cafeteria Attendants and Bartender Helpers',
                  'Dishwashers',
                  'Models',
                  'Singers',
                  'Cashiers',
                  'Shampooers',
                  'Telemarketers',
                  'Door-To-Door Sales Workers, News and Street Vendors, and Related Workers',
                  'Tapers',
                  'Fishers and Related Fishing Workers',
                  'Farmworkers and Laborers, Crop',
                  'Sewers, Hand',
                  'Textile Bleaching and Dyeing Machine Operators and Tenders',
                  'Textile Cutting Machine Setters, Operators, and Tenders',
                  'Textile Knitting and Weaving Machine Setters, Operators, and Tenders',
                  'Textile Winding, Twisting, and Drawing Out Machine Setters, Operators, and Tenders',
                  'Shoe Machine Operators and Tenders',
                  'Shoe and Leather Workers and Repairers',
                  'Sewing Machine Operators',
                  'Printing Press Operators',
                  'Print Binding and Finishing Workers',
                  'Laundry and Dry-Cleaning Workers',
                  'Bakers',
                  'Butchers and Meat Cutters',
                  'Meat, Poultry, and Fish Cutters and Trimmers',
                  'Slaughterers and Meat Packers',
                  'Food and Tobacco Roasting, Baking, and Drying Machine Operators and Tenders',
                  'Food Batchmakers',
                  'Food Cooking Machine Operators and Tenders']))

knowledge = [] # seems not helpful
             
skills = ['Critical Thinking', 
          'Complex Problem Solving',
          'Active Learning',
          'Judgment and Decision Making',
          'Reading Comprehension']

abilities = ['Written Comprehension',
             'Written Expression',
             'Fluency of Ideas', #the number of ideas is important, not their quality, correctness, or creativity
             'Originality',
             'Problem Sensitivity',
             'Deductive Reasoning',
             'Inductive Reasoning',
             'Information Ordering',
             'Category Flexibility',
             'Mathematical Reasoning',
             'Static Strength',
             'Stamina'
             ]

interests = ['Investigative']

work_style = ['Attention to Detail',
              'Analytical Thinking',
              'Innovation']

work_activities = ['Analyzing Data or Information',
                   'Making Decisions and Solving Problems',
                   'Thinking Creatively']


df['SOC']= df['O*NET-SOC Code'].str[0:7]

df['SOC'].nunique()


dt = pd.read_excel('db_24_1_excel/Knowledge.xlsx')
dt['SOC']= df['O*NET-SOC Code'].str[0:7]

dt['SOC'].nunique()

work_style = pd.read_excel('db_24_1_excel/Work Styles.xlsx')

work_style = work_style[['O*NET-SOC Code', 'Title', 'Element Name', 'Data Value']]

work_st=work_style[['Title', 'Element Name', 'Data Value']].pivot(index='Title',
                 columns='Element Name',
                 values='Data Value')

dd=work_style.set_index(['O*NET-SOC Code', 'Title', 'Element Name']).unstack('Element Name')

