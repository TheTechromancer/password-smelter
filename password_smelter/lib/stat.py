# by TheTechromancer

import re
import sys
import numpy as np
import pandas as pd
from math import log, e as euler
from collections import OrderedDict
from password_stretcher.lib.policy import PasswordPolicy

# graphing
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class Splitter:

    types = [
            re.compile(r'^-{0,1}\d+$'),
            re.compile(r'^\d+-$'),
            re.compile(r'^\d+-\d+$'),
        ]

    def __init__(self, delimiter, field):

        self.delimiter = str(delimiter)

        self.field1 = 0
        self.field2 = 0
        try:
            self.field1 = int(field)
        except ValueError:
            try:
                field1, field2 = field.split('-', 1)
                self.field1 = int(field1)
                self.field2 = int(field2)
            except ValueError:
                pass

        if self.field1 > 0:
            self.field1 -= 1
        if self.field2 > 0:
            self.field2 -= 1

        self.split_type = self.get_split_type(field)
        if self.split_type == -1:
            assert False, f'Invalid field: {field}'


    def get_split_type(self, f):
        for i,t in enumerate(self.types):
            if t.match(f):
                return i
        return -1


    def split(self, s):

        if self.split_type == 0:
            return s.split(self.delimiter)[self.field1]
        elif self.split_type == 1:
            return self.delimiter.join(s.split(self.delimiter)[self.field1:])
        elif self.split_type == 2:
            return self.delimiter.join(s.split(self.delimiter)[self.field1:self.field2])




class Stat:

    # TODO: make unicode-friendly
    word_regex = re.compile(r'[a-z][a-z01357$@]+[a-z]', re.I)
    number_regex = re.compile(r'\d+')
    symbol_regex = re.compile(r'[^a-z0-9]+', re.I)

    # sort before truncation
    presort = {'by': 1, 'ascending': False}
    # sort after truncation
    postsort = {'by': 1, 'ascending': False}

    hiderare = True
    hidesingles = True
    include_other = True

    def __init__(self, options, title=None, key_label=None, val_label=None):

        self.title = ('' if title is None else str(title))
        self.key_label = ('' if key_label is None else str(key_label))
        self.val_label = ('' if val_label is None else str(val_label))

        self.options = options
        self._json = dict()

        # caching
        self._df = None


    @property
    def json(self):

        return self._json


    def increment(self, k):
        '''
        override if needed
        '''

        self._increment(k)


    def make_figure(self, theme):

        if not self.df.empty:
            return self.make_bar(title=self.title, theme=theme)


    def make_bar(self, title=None, theme=None):

        #fig = px.bar(df, title=title, x=df.columns[0], y=df.columns[1], template=theme)
        fig = go.Figure(data=[self._make_bar()])
        fig.update_layout(
            title_text=title,
            template=theme,
            font=dict(size=20),
            legend={
                'bordercolor': 'rgba(255, 255, 255, 0)'
            }
        )

        fig.update_xaxes(type='category')
        return fig


    def _make_bar(self, name=None):

        if name is None:
            name = self.title

        return go.Bar(x=self.df.iloc[:, 0], y=self.df.iloc[:, 1], name=name)


    def _make_scatter(self, name=None):

        if name is None:
            name = self.title

        return go.Scatter(x=self.df.iloc[:, 0], y=self.df.iloc[:, 1], name=name, mode='markers')


    @staticmethod
    def make_pie(df, title=None, theme=None):

        #fig = px.pie(df, title=title, names=df.columns[0], values=df.columns[1])
        fig = go.Figure(data=[go.Pie(labels=df.iloc[:, 0], values=df.iloc[:, 1], hole=.3)])
        fig.update_layout(title_text=title, height=650, template=theme, font=dict(size=20))
        return fig


    def _increment(self, k):

        try:
            self._json[k] += 1
        except KeyError:
            self._json[k] = 1


    @property
    def df(self):
        '''
        returns self as pandas data frame
        '''

        if self._df is None:

            if self.title.strip():
                print(f'[+] Calculating {self.title}')

            df = pd.DataFrame(self.json.items())
            if not df.empty:
                # sort before dropping takes place
                df.sort_values(**self.presort, ignore_index=True, inplace=True)

                # combine rare and truncated data points
                rare = (
                    df[
                        (df[1] < df[1].max() * (self.options.hiderare / 100) if self.hiderare else df[1] == '_eyes_') | \
                        (df[1] == 1 if self.hidesingles else df[1] == '_throat_') | \
                        (df.index >= self.options.limit if True else df[1] == '_genitals_')
                    ]
                )

                '''
                if self.hiderare:
                    hiderare_threshold = df[1].max() * (self.hiderare / 100)
                    rare = df[(df[1] < hiderare_threshold) | (df.index >= self.options.limit)]
                else:
                    rare = df[df.index >= self.options.limit]

                if self.hidesingles:
                    hiddensingles = df[df[1] == 1]
                    rare = pd.concat([rare, hiddensingles])
                '''

                df.drop(rare.index, inplace=True)

                # sort after dropping has taken place
                df.sort_values(**self.postsort, ignore_index=True, inplace=True)

                # check if there's already an 'Other' category
                if self.include_other:
                    other = df[df[0]=='Other']
                    if other.empty:
                        df = df.append(pd.DataFrame([('Other', rare[1].sum())]), ignore_index=True)
                    # if so, add to it instead
                    else:
                        df.loc[df[0]=='Other',1] += rare[1].sum()

                # rename labels
                df.rename(columns={0: self.key_label, 1: self.val_label}, inplace=True)

            self._df = df

        return self._df





class CharacterSetStat(Stat):

    @property
    def json(self):

        return {PasswordPolicy.charset_flags[k]: v for k,v in self._json.items()}


class PasswordStat(Stat):

    hiderare = False
    include_other = False


class MutationStat(Stat):

    include_other = False

    def increment(self, password):

        for match in self.word_regex.finditer(password):
            mutation = f'{password[:match.span()[0]]}password{password[match.span()[1]:]}'
            super().increment(mutation)


class PasswordLengthStat(Stat):

    postsort = {'by': 0}



class PasswordEntropyStat(Stat):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self._json['coordinates'] = dict()


    def increment(self, password):

        try:
            self._json['coordinates'][(self.entropy(password), len(password))] += 1
        except KeyError:
            self._json['coordinates'][(self.entropy(password), len(password))] = 1


    def _make_scatter(self, *args, name=None, **kwargs):

        if name is None:
            name = self.title

        size = self.df.iloc[:, 2]
        fig = go.Scatter(
            x=self.df.iloc[:, 0],
            y=self.df.iloc[:, 1],
            mode='markers',
            marker={
                'size': size,
                'sizemode': 'area',
                'sizeref': 2.*max(size)/(40.**2),
                'sizemin': 3
            },
            name=name
        )

        return fig


    @property
    def df(self):

        if self._df is None:

            print(f'[+] Calculating {self.title}')

            df = pd.DataFrame([(a, b, c) for ((a,b) ,c) in self._json['coordinates'].items()])
            if not df.empty:
                df.sort_values(**{'by': 2, 'ascending': False}, ignore_index=True, inplace=True)
                df.drop(df[df[0] == 0].index, inplace=True)
                df.rename(columns={0: self.key_label, 1: self.val_label, 2: 'Count'}, inplace=True)
            self._df = df

        return self._df


    @staticmethod
    def entropy(password, base=None):

        password = list(password)
        n_labels = len(password)

        if n_labels <= 1:
            return 0

        value,counts = np.unique(password, return_counts=True)
        probs = counts / n_labels
        n_classes = np.count_nonzero(probs)

        if n_classes <= 1:
            return 0

        ent = 0.

        # Compute entropy
        base = euler if base is None else base
        for i in probs:
            ent -= i * log(i, base)

        return ent



class PasswordMetaStat(Stat):

    def __init__(self, cracked, uncracked, complex_count, noncomplex_count, complex_length, noncomplex_length, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.cracked = cracked
        self.uncracked = uncracked
        self.complex_count = complex_count
        self.noncomplex_count = noncomplex_count
        self.complex_length = complex_length
        self.noncomplex_length = noncomplex_length
        self.total = cracked + uncracked


    def make_figure(self, *args, **kwargs):

        return self.make_sunburst(*args, **kwargs)


    def make_sankey(self, theme):

        fig = go.Figure(data=[go.Sankey(
            valueformat = ".0f",
            valuesuffix = "TWh",
            # Define nodes
            node = dict(
              pad = 15,
              thickness = 15,
              line = dict(color = "black", width = 0.5),
              label =  data['data'][0]['node']['label'],
              color =  data['data'][0]['node']['color']
            ),
            # Add links
            link = dict(
              source =  data['data'][0]['link']['source'],
              target =  data['data'][0]['link']['target'],
              value =  data['data'][0]['link']['value'],
              label =  data['data'][0]['link']['label'],
              color =  data['data'][0]['link']['color']
        ))])
        return fig


    def make_sunburst(self, theme):

        root = f'{self.total:,}'

        labels = [root]
        values = ['']
        parents = ['']

        # cracked layer
        if self.uncracked:
            labels += ['Cracked', 'Uncracked']
            values += [self.cracked, self.uncracked]
            parents += [root] * 2

        if self.noncomplex_count > 0:
            # complexity layer
            labels += ['Compliant', 'Non-Compliant']
            values += [self.complex_count, self.noncomplex_count]
            if self.uncracked:
                parents += [f'Cracked'] * 2
            else:
                parents += [root] * 2

        # length layers
        '''
        labels += [str(_) for _ in self.complex_length.df.iloc[:, 0]]
        values += list(self.complex_length.df.iloc[:, 1])
        parents += ['Meets Complexity'] * len(self.complex_length.df.iloc[:, 1])

        labels += [str(_) for _ in self.noncomplex_length.df.iloc[:, 0]]
        values += list(self.noncomplex_length.df.iloc[:, 1])
        parents += ['Fails Complexity'] * len(self.noncomplex_length.df.iloc[:, 1])
        '''

        fig =px.sunburst(
            {
                'labels': labels,
                'values': values,
                'parents': parents,
            },
            names='labels',
            values='values',
            parents='parents',
            branchvalues='total',
            color='labels',
            color_discrete_sequence=px.colors.qualitative.Plotly,
            color_discrete_map={
                'Cracked': px.colors.qualitative.Plotly[1],
                'Uncracked': px.colors.qualitative.Plotly[0],
                'Compliant': px.colors.qualitative.Plotly[0],
                'Non-Compliant': px.colors.qualitative.Plotly[1],
            },
        )

        fig.update_layout(
            height=600,
            template=theme,
            margin=dict(t=0, l=0, r=0, b=0),
            font_color='white',
            font=dict(size=20),
            title_font_color='white',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )


        '''
        fig =go.Figure(go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues='total',
            sort=False,
            #color_discrete_map={'Cracked': 'red', 'Uncracked': 'green'}
        ))

        #fig.update_xaxes(tickfont=dict(color='rgba(0,0,0,0)'))
        #fig.update_yaxes(tickfont=dict(color='rgba(0,0,0,0)'))
        '''

        return fig



class AdvancedMaskStat(Stat):

    def make_figure(self, theme):

        import plotly.figure_factory as ff

        fig = make_subplots(
            rows=1, cols=2,
            specs=[
                [{'type': 'table'}, {'type': 'pie'}]
            ]
        )

        fig.add_trace(
            go.Table(
                header={
                    'values': list(self.df.columns),
                },
                cells={
                    'values': [self.df[k].tolist() for k in self.df.columns]
                }
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Pie(
                labels=self.df.iloc[:, 0],
                values=self.df.iloc[:, 1],
                hole=.3
            ),
            row=1, col=2
        )

        fig.update_layout(title_text=self.title, height=650, template=theme)

        return fig




class BaseWordStat(Stat):

    # this should work but it doesn't
    # re.compile(r'[^\d\W]([01357@$]|[^\d\W])+[^\d\W]').findall('Pa$$wórd123')
    hiderare = False
    include_other = False

    def increment(self, k):

        words = []
        for word in self.word_regex.findall(k):
            word = word.lower()
            if not word in words:
                words.append(word)
        for word in words:
            self._increment(word)



class NumberStat(Stat):

    # this should work but it doesn't
    # re.compile(r'[^\d\W]([01357@$]|[^\d\W])+[^\d\W]').findall('Pa$$wórd123')
    hiderare = False
    include_other = False

    def increment(self, k):

        numbers = []
        for number in self.number_regex.findall(k):
            if not number in numbers:
                numbers.append(number)
        for number in numbers:
            self._increment(number)



class SymbolStat(Stat):

    # this should work but it doesn't
    # re.compile(r'[^\d\W]([01357@$]|[^\d\W])+[^\d\W]').findall('Pa$$wórd123')
    hiderare = False
    include_other = False

    def increment(self, k):

        symbols = []
        for symbol in self.symbol_regex.findall(k):
            if not symbol in symbols:
                symbols.append(symbol)
        for symbol in symbols:
            self._increment(symbol)




class PasswordStats:

    def __init__(self, options):

        self.options = options
        self.splitter = Splitter(options.delimiter, options.field)

        # complexity filters
        self.policy = PasswordPolicy(
            minlength=options.minlength,
            maxlength=options.maxlength,
            mincharsets=options.mincharsets,
            required_charsets=options.charsets,
            regex=options.regex,
        )

        # global password stats
        self._meta = None

        self.overall = OrderedDict([
            ('passwords', PasswordStat(self.options, title='Top Passwords', key_label='Password', val_label='Count')),
            ('basewords', BaseWordStat(self.options, title='Top Base Words', key_label='Base Word', val_label='Count')),
            ('mutations', MutationStat(self.options, title='Top Mutations', key_label='Mutation', val_label='Count')),
            ('numbers', NumberStat(self.options, title='Top Numbers', key_label='Number', val_label='Count')),
            ('symbols', SymbolStat(self.options, title='Top Symbols', key_label='Symbol', val_label='Count')),
            ('advancedmasks', AdvancedMaskStat(self.options, title='Advanced Masks', key_label='Mask', val_label='Count')),
        ])

        # passwords that meet complexity filters
        self.complex_counter = 0
        self.complex = OrderedDict([
            ('length', PasswordLengthStat(self.options, title='Password Length', key_label='Password Length', val_label='Count')),
            ('entropy', PasswordEntropyStat(self.options, title='Password Entropy', key_label='Entropy', val_label='Password Length')),
            ('simplemasks', Stat(self.options, title='Simple Masks', key_label='Mask', val_label='Count')),
            ('charactersets', CharacterSetStat(self.options, title='Character Sets', key_label='Character Set', val_label='Count')),
        ])

        # passwords that don't meet complexity filters
        self.noncomplex_counter = 0
        self.noncomplex = OrderedDict([
            ('length', PasswordLengthStat(self.options, title='Password Length', key_label='Password Length', val_label='Count')),
            ('entropy', PasswordEntropyStat(self.options, title='Password Entropy', key_label='Password Entropy', val_label='Password Length')),
            ('simplemasks', Stat(self.options, title='Simple Masks', key_label='Mask', val_label='Count')),
            ('charactersets', CharacterSetStat(self.options, title='Character Sets', key_label='Character Set', val_label='Count')),
        ])

        self.cracked = 0
        self.uncracked = 0


    def analyze_password(self, password):

        meets_policy, pass_length, charset, num_charsets, simplemask, advancedmask = self.policy.analyze_password(password)

        if meets_policy:
            self.complex_counter += 1
            stats_category = self.complex
        else:
            self.noncomplex_counter += 1
            stats_category = self.noncomplex

        self.overall['passwords'].increment(password)
        self.overall['basewords'].increment(password)
        self.overall['mutations'].increment(password)
        self.overall['numbers'].increment(password)
        self.overall['symbols'].increment(password)
        self.overall['advancedmasks'].increment(advancedmask)
        stats_category['length'].increment(pass_length)
        stats_category['entropy'].increment(password)
        stats_category['simplemasks'].increment(simplemask)
        stats_category['charactersets'].increment(charset)


    @property
    def meta(self):

        return PasswordMetaStat(
            self.cracked,
            self.uncracked,
            self.complex_counter,
            self.noncomplex_counter,
            self.complex['length'],
            self.noncomplex['length'],
            options=self.options
        )
    

    def analyze(self, iterator):
        ''' Generate password statistics. '''

        for line in iterator:
            if self.options.delimiter:
                try:
                    password = self.splitter.split(line)
                    print(password)
                except IndexError:
                    self.uncracked += 1
                    continue
            else:
                password = line

            if len(password) > 0:
                self.cracked += 1
                self.analyze_password(password)
            elif self.options.delimiter:
                self.uncracked += 1

            sys.stdout.write(f'\r[+] Read {self.cracked:,} passwords')

        print('')


    def __getattr__(self, attr):

        return self.complex[attr]


    def __iter__(self):

        for k,stat in self.overall.items():
            yield k,stat
        for k,stat in self.complex.items():
            yield k,stat
        for k,stat in self.noncomplex.items():
            yield k,stat