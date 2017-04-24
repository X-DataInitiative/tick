import os
from sphinx.directives.other import Include as SphinxInclude
from docutils.parsers.rst import directives


class Include(SphinxInclude):
    """Custom directive to add an "allowmissing" flag to the include
    directive.
    This avoid errors if we want to include a file that might not exists.
    """
    option_spec = {
        'allowmissing': directives.flag
    }

    def run(self):
        if "allowmissing" in self.options:
            env = self.state.document.settings.env
            rel_filename, filename = env.relfn2path(self.arguments[0])
            if os.path.exists(filename):
                return SphinxInclude.run(self)
            else:
                return []
        else:
            return SphinxInclude.run(self)


def setup(app):
    directives.register_directive('include', Include)
