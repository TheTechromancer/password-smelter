# by TheTechromancer

import json
from .stat import *
from pathlib import Path
from threading import Thread
from datetime import datetime
from collections import OrderedDict
from password_stretcher.lib.utils import *
from password_stretcher.lib.errors import *

# graphing
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class PasswordReport:

    def __init__(self, options):

        self.options = options
        self.stats = PasswordStats(self.options)


    @property
    def plotly_theme(self):

        if self.options.theme == 'light':
            return 'plotly'
        elif self.options.theme == 'dark':
            return 'plotly_dark'


    @property
    def dash_theme(self):

        import dash_bootstrap_components as dbc

        if self.options.theme == 'light':
            return dbc.themes.FLATLY
        elif self.options.theme == 'dark':
            return dbc.themes.CYBORG


    @property
    def background_color(self):

        return 'rgba(0,0,0,0)'
        if self.options.theme == 'light':
            return '#fff'
        elif self.options.theme == 'dark':
            return '#000'


    @property
    def grid_color(self):

        return 'rgba(128,128,128,.3)'


    def dump_everything(self, dirname, write=True):

        date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        markdown = [f'# {self.options.title}']
        json_data = dict()

        dirname = Path(str(dirname))
        dirname.mkdir(exist_ok=True, parents=True)

        if write:
            print(f'[+] Outputting passwords to {dirname}')

            assert dirname.is_dir(), f'Problem with output directory: {dirname}'

            # excel
            excel_filename = f'{date_str}_password_analysis.xlsx'
            self.make_excel_report(dirname / excel_filename)

        # png
        figures = self.make_html_report(show=False)
        fig_dirname = f'{date_str}_password_analysis_images'
        if write:
            (dirname / fig_dirname).mkdir(exist_ok=True, parents=True)
            assert (dirname / fig_dirname).is_dir(), f'Problem with output directory: {fig_dirname}'

        kaleido_error = False
        for title, (fig, stat) in figures.items():
            if not stat.df.empty:
                fig_filename = f'{title}.png'
                if title not in ['entropy']:
                    markdown.append(f'## {stat.title}')
                    if write:
                        markdown.append(f'![{stat.title}]({Path(fig_dirname) / fig_filename})')
                    markdown.append(stat.df.to_markdown(index=False))
                json_data[title] = stat.df.to_dict()
                if write:
                    with open(dirname / fig_dirname / fig_filename, 'wb') as f:
                        try:
                            fig.write_image(f, width=1200, height=500, scale=2, engine="kaleido")
                        except ValueError as e:
                            if not kaleido_error:
                                print(f'[!] {str(e).lstrip()}')
                                kaleido_error = True

        if write:
            json_filename = dirname / f'{date_str}_password_analysis.json'
            with open(json_filename, 'w') as f:
                json.dump(json_data, f, indent=4)

        markdown = '\n\n'.join(markdown)

        if write:
            markdown_filename = dirname / f'{date_str}_password_analysis.md'
            with open(markdown_filename, 'w') as f:
                f.write(markdown)

        return markdown


    def make_excel_report(self, filename):

        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter(str(filename), engine='xlsxwriter')

        # Write each dataframe to a different worksheet.
        for k,stat in self.stats:
            stat.df.to_excel(writer, sheet_name=str(stat.title), index=False)

        # Close the Pandas Excel writer and output the Excel file.
        writer.save()


    def make_html_report(self, show=True):

        print('[+] Generating report')

        import dash
        import dash_core_components as dcc
        import dash_html_components as html

        if self.stats.cracked <= 0:
            assert False, 'No passwords to analyze'

        '''
        fig = make_subplots(
            rows=1, cols=2,
            specs=[
                [{'type': 'bar'}, {'type': 'bar'}]
            ]
        )

        fig.add_trace(
            self.stats.complex['length']._make_bar(name='Compliant'),
            row=1, col=1
        )
        fig.add_trace(
            self.stats.noncomplex['length']._make_bar(name='Non-Compliant'),
            row=1, col=2
        )
        #fig.add_trace(self.stats.noncomplex['length'].make_figure(theme=self.plotly_theme), row=1, col=2)
        '''

        figures = OrderedDict()
        graphs = []

        if self.stats.uncracked or self.stats.policy:
            fig = self.stats.meta.make_figure(theme=self.plotly_theme)
            figures['overall'] = (fig, self.stats.meta)
            graphs.append(dcc.Graph(figure=fig))

        for k in ['passwords', 'basewords', 'mutations', 'numbers', 'symbols']:
            stat = self.stats.overall[k]
            fig = stat.make_figure(theme=self.plotly_theme)
            figures[k] = (fig, stat)

        # everything else
        for k,stat in [i for i in self.stats.complex.items() if i[0] != 'entropy']:
            if self.stats.noncomplex_counter > 0:
                figure_data = [
                    self.stats.complex[k]._make_bar(name='Compliant'),
                    self.stats.noncomplex[k]._make_bar(name='Non-Compliant'),
                ]
            else:
                figure_data = [
                    self.stats.complex[k]._make_bar()
                ]
            fig = go.Figure(data=figure_data)
            fig.update_layout(
                font=dict(size=20),
                title_text=stat.title,
                template=self.plotly_theme,
                barmode='group',
                xaxis={'categoryorder': 'trace'}
            )
            fig.update_xaxes(type='category')
            figures[k] = (fig, stat)

        # password entropy
        for k,stat in [('entropy', self.stats.complex['entropy'])]:
            if self.stats.noncomplex_counter > 0:
                figure_data = [
                    self.stats.complex[k]._make_scatter(name='Compliant'),
                    self.stats.noncomplex[k]._make_scatter(name='Non-Compliant'),
                ]
            else:
                figure_data = [
                    self.stats.complex[k]._make_scatter()
                ]
            fig = go.Figure(data=figure_data)
            fig.update_layout(
                font=dict(size=20),
                title_text=stat.title,
                template=self.plotly_theme,
                xaxis={
                    'title': stat.key_label,
                },
                yaxis={
                    'title': stat.val_label,
                },
            )
            figures[k] = (fig, stat)

        figures['advancedmasks'] = (
            self.stats.overall['advancedmasks'].make_figure(theme=self.plotly_theme),
            self.stats.overall['advancedmasks']
        )

        for k, (fig, stat) in figures.items():
            if not stat.df.empty:
                graphs.append(dcc.Graph(figure=fig))

        for graph in graphs:
            graph.figure.layout.plot_bgcolor = self.background_color
            graph.figure.layout.paper_bgcolor = self.background_color
            graph.figure.layout.xaxis.gridcolor = self.grid_color
            graph.figure.layout.yaxis.gridcolor = self.grid_color

        if show:
            app = dash.Dash(__name__, external_stylesheets=[self.dash_theme])
            app.layout = html.Div(
                style={
                    'padding': '1.5em',
                    'max-width': '100em',
                    'margin': '0 auto',
                },
                children=[
                    html.H1(
                        style={
                            'text-align': 'center'
                        },
                        children=self.options.title
                    ),
                    html.H5(
                        style={
                            'color': '#999',
                            'text-align': 'center',
                            'padding-bottom': '2em'
                        },
                        children=f'{self.stats.cracked:,} passwords ({self.stats.complex_counter/self.stats.cracked*100:.1f}% compliant)'
                    ),
                ] + graphs
            )
            try:
                Thread(target=thread_wrapper, args=(app.run_server,), kwargs=dict(host=self.options.host, port=self.options.port)).start()
            except Exception as e:
                raise PasswordAnalyzerError(e)

            if not self.options.no_browser:
                if self.options.host == '0.0.0.0':
                    host = '127.0.0.1'
                else:
                    host = str(self.options.host)
                import webbrowser
                webbrowser.open(f'http://{host}:{self.options.port}')

        return figures