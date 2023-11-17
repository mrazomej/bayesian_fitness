function Div(el)
    -- Check if the div has a specific class, e.g., 'maintext'
    if el.classes:includes("sitext") then
        -- Define multi-line string for the beginning of the custom LaTeX block
        local sitextStart = [[
% SUPPLEMENTAL MATERIAL

% Indicate that all sections and subsections should be included in the
% table of contents so that only the SI is included.
\addtocontents{toc}{\protect\setcounter{tocdepth}{3}}

% Define the reference section for the supplemental material
\begin{refsegment}
% Set equation, table, and figure counters to begin with "S"
\beginsupplement

%% OPTIONAL: Add title for section if it doesn't exist
\hypertarget{supplementary-materials}{%
\section{Supplementary Materials}\label{supplementary-materials}}

% Add table of contents
\tableofcontents
]]

        -- Define multi-line string for the end of the custom LaTeX block
        local sitextEnd = [[
% Print supplemental references changing the title
\printbibliography[title={Supplemental References},
segment=\therefsegment, filter=notother]
\end{refsegment}
]]

        -- Convert the div to a custom LaTeX block
        local latexStart = pandoc.RawBlock('latex', sitextStart)
        local latexEnd = pandoc.RawBlock('latex', sitextEnd)

        -- Insert the LaTeX start and end commands around the div content
        table.insert(el.content, 1, latexStart)
        table.insert(el.content, latexEnd)

        return el.content
    end
end
