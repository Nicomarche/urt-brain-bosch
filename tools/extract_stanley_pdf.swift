import Foundation
import PDFKit

struct PDFStanleyExtractor {
    let pdfPath: String
    let keywords: [String]

    func run() -> Int32 {
        let url = URL(fileURLWithPath: pdfPath)
        guard FileManager.default.fileExists(atPath: url.path) else {
            fputs("ERROR: PDF file not found at path: \(url.path)\n", stderr)
            return 1
        }

        guard let document = PDFDocument(url: url) else {
            fputs("ERROR: Could not open PDF document.\n", stderr)
            return 1
        }

        print("PDF: \(url.path)")
        print("Pages: \(document.pageCount)")
        print("")

        var hitCount = 0
        let normalizedKeywords = keywords.map { $0.lowercased() }

        for pageIndex in 0..<document.pageCount {
            guard let page = document.page(at: pageIndex),
                  let pageText = page.string else {
                continue
            }

            let lines = pageText
                .components(separatedBy: .newlines)
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }

            var pageHasHits = false
            var pageHits: [String] = []

            for line in lines {
                let lower = line.lowercased()
                if normalizedKeywords.contains(where: { lower.contains($0) }) {
                    pageHasHits = true
                    pageHits.append(line)
                    hitCount += 1
                }
            }

            if pageHasHits {
                print("=== Page \(pageIndex + 1) ===")
                for line in pageHits {
                    print("- \(line)")
                }
                print("")
            }
        }

        print("Total keyword hits: \(hitCount)")
        return 0
    }
}

func main() -> Int32 {
    var args = CommandLine.arguments
    args.removeFirst()

    guard !args.isEmpty else {
        print("Usage: swift tools/extract_stanley_pdf.swift /path/to/file.pdf")
        return 1
    }

    let pdfPath = args[0]
    let keywords = [
        "stanley",
        "control law",
        "controller",
        "cross-track",
        "crosstrack",
        "heading",
        "yaw",
        "steering",
        "atan",
        "arctan",
        "speed",
        "velocity",
        "gain",
        "error",
        "k",
        "psi",
        "delta"
    ]

    let extractor = PDFStanleyExtractor(pdfPath: pdfPath, keywords: keywords)
    return extractor.run()
}

exit(main())
