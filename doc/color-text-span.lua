function Span(el)
    -- Check if the 'style' attribute exists and contains 'color'
    if el.attributes.style and string.find(el.attributes.style, "color") then
        -- Extract the color value from the style string (either a name or a hex code)
        local color = string.match(el.attributes.style, "color:%s*([^;]+)")

        -- If a color is found and the target format is LaTeX
        if color and FORMAT:match 'latex' then
            local latexColorCmd

            -- If the color is in hexadecimal format (e.g., #FFFFFF), use \definecolor
            if color:sub(1, 1) == "#" then
                -- Remove the '#' and define the color
                local hex = color:sub(2)
                latexColorCmd = '\\definecolor{tempcolor}{HTML}{' .. hex .. '}\\textcolor{tempcolor}'
            else
                -- For named colors, use them directly
                latexColorCmd = '\\textcolor{' .. color .. '}'
            end

            -- Wrap the content of the span in the appropriate LaTeX \textcolor command
            local latexStart = pandoc.RawInline('latex', latexColorCmd .. '{')
            local latexEnd = pandoc.RawInline('latex', '}')

            -- Insert the LaTeX commands before and after the original content
            table.insert(el.content, 1, latexStart)
            table.insert(el.content, latexEnd)

            -- Return the modified content
            return el.content
        end
    end

    -- For other cases or formats, return the element unchanged
    return el
end
