function Div(el)
    -- Check if the div has the 'maintext' class
    if el.classes:includes("maintext") then
        -- Apply modifications if the output format is PDF
        if FORMAT:match("latex") then
            -- Define multi-line string for the beginning of the custom LaTeX block
            local maintextStart = [[
% Remove main text from the table of contents by specifying not to include
% any section or subsection
\addtocontents{toc}{\protect\setcounter{tocdepth}{-1}}

% Define reference segment for main text
\begin{refsegment}
% Generate filter to not include references from main text in the
% supplemental references
\defbibfilter{notother}{not segment=\therefsegment}
]]

            -- Define multi-line string for the end of the custom LaTeX block
            local maintextEnd = [[
% Print main text references
\printbibliography[segment=\therefsegment]
% Close reference segment
\end{refsegment}

\clearpage
]]

            -- Convert the div to a custom LaTeX block
            local latexStart = pandoc.RawBlock('latex', maintextStart)
            local latexEnd = pandoc.RawBlock('latex', maintextEnd)

            -- Insert the LaTeX start and end commands around the div content
            table.insert(el.content, 1, latexStart)
            table.insert(el.content, latexEnd)

            return el.content

            -- Apply different modifications for HTML output
        elseif FORMAT:match("html") then
            -- For HTML, return the content only (ignoring the div wrapper)
            return el.content
        end
    end

    -- For divs without the 'maintext' class or other formats, return the div unchanged
    return el
end
